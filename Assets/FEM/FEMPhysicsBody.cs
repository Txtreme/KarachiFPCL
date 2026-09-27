using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class FEMPhysicsBody : MonoBehaviour
{
    // Active bodies registry for GPU Inter-Body Collisions
    private static readonly List<FEMPhysicsBody> allActiveBodies = new List<FEMPhysicsBody>();

    [Header("Source Mesh Configuration")]
    [Tooltip("Assign the source 3D volume mesh (e.g. Sphere, Cube, FBX). If left empty, auto-detects from MeshFilter on Awake.")]
    public Mesh sourceMesh;

    [Header("Compute Shader Reference")]
    public ComputeShader femComputeShader;

    [Header("Tetrahedrization & Element Size")]
    public float targetTetSize = 0.05f;
    public Vector3Int gridResolution = new Vector3Int(8, 8, 8);

    [Header("FEM Physical Parameters")]
    [Range(1, 30)] public int solverSubsteps = 5;
    [Range(1, 10)] public int solverIterations = 2;
    public float particleMass = 0.1f;
    public Vector3 gravity = new Vector3(0, -9.81f, 0);
    [Range(0f, 1f)] public float damping = 0.02f;
    [Header("Self Collision")]
    public bool enableSelfCollision = true;
    public float particleRadius = 0.025f;
    [Header("Material Elasticity")]
    public float youngsModulus = 5000f;
    [Range(0.0f, 0.49f)] public float poissonsRatio = 0.45f;

    public bool enableFloorCollision = true;
    public float floorY = 0f;

    [Header("Debug Visualization")]
    public bool drawTetWireframe = true;

    private TetrahedralMesh tetMesh;
    private Vector3[] positions;
    private Vector3[] prevPositions;
    private Vector3[] velocities;
    private float[] invMasses;
    private Vector3[] restPositions;
    private Vector3[] localVertsCache;
    private int[] cachedSurfaceTriangles;

    public TetrahedralMesh TetMesh => tetMesh;
    public Vector3[] CurrentPositions => positions;
    public Vector3[] RestPositions => restPositions;

    private GPUMatrix3x3[] invRestMatrices;
    public Vector3[] Velocities => velocities;

    // GPU Compute Buffers
    private ComputeBuffer positionsBuffer;
    private ComputeBuffer prevPositionsBuffer;
    private ComputeBuffer velocitiesBuffer;
    private ComputeBuffer invMassesBuffer;
    private ComputeBuffer tetsBuffer;
    private ComputeBuffer invRestMatricesBuffer;
    private ComputeBuffer deltaPosIntBuffer;
    private ComputeBuffer deltaCountBuffer;

    public ComputeBuffer PositionsBuffer => positionsBuffer;

    private int kIntegrate, kSolveFEM, kApplyDeltas, kPostPhysics, kParticleCollisions, kInterBodyCollisions;

    private bool isInitialized = false;
    private FEMVisualShell cachedVisualShell;
    private Mesh generatedColliderMesh;

    [StructLayout(LayoutKind.Sequential)]
    public struct GPUTetrahedron
    {
        public int v0, v1, v2, v3;
        public float restVolume;
        public int active;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct GPUMatrix3x3
    {
        public float m00, m01, m02;
        public float m10, m11, m12;
        public float m20, m21, m22;
    }

    private void Awake()
    {
        EnsureComponentsExist();
        cachedVisualShell = GetComponent<FEMVisualShell>();

        if (sourceMesh == null)
        {
            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf != null && mf.sharedMesh != null)
            {
                if (!mf.sharedMesh.name.Contains("FEM_Visual_Shell") && !mf.sharedMesh.name.Contains("FEM_VisualMesh"))
                {
                    sourceMesh = mf.sharedMesh;
                }
            }
        }
    }

    private void Start()
    {
        if (isInitialized) return;

        if (tetMesh == null)
        {
            if (sourceMesh == null)
            {
                Debug.LogError("FEMPhysicsBody: No source Mesh assigned or found on MeshFilter!");
                return;
            }

            if (!sourceMesh.isReadable)
            {
                Debug.LogError($"FEMPhysicsBody: Mesh '{sourceMesh.name}' is not Read/Write enabled!");
                return;
            }

            Vector3 boundsSize = sourceMesh.bounds.size;
            if (boundsSize.x * boundsSize.y * boundsSize.z <= 1e-6f)
            {
                Debug.LogError($"FEMPhysicsBody: Mesh '{sourceMesh.name}' has no 3D volume. FEM requires a watertight 3D mesh.");
                return;
            }

            if (targetTetSize > 0.001f)
            {
                gridResolution = new Vector3Int(
                    Mathf.Max(2, Mathf.CeilToInt(boundsSize.x / targetTetSize)),
                    Mathf.Max(2, Mathf.CeilToInt(boundsSize.y / targetTetSize)),
                    Mathf.Max(2, Mathf.CeilToInt(boundsSize.z / targetTetSize))
                );
            }

            tetMesh = TetrahedralMesh.GenerateFromMesh(sourceMesh, gridResolution);
        }

        if (tetMesh == null || tetMesh.vertices == null || tetMesh.vertices.Count == 0 || tetMesh.tets == null || tetMesh.tets.Count == 0)
        {
            Debug.LogError("FEMPhysicsBody: Tetrahedral generation returned 0 elements.");
            return;
        }

        InitializePhysicsState();
    }

    private void OnEnable()
    {
        MeshCutter.RegisterBody(this);
        if (!allActiveBodies.Contains(this))
        {
            allActiveBodies.Add(this);
        }
    }

    private void OnDisable()
    {
        MeshCutter.UnregisterBody(this);
        allActiveBodies.Remove(this);
        ReleaseGPUResources();
    }

    private void EnsureComponentsExist()
    {
        if (GetComponent<MeshFilter>() == null) gameObject.AddComponent<MeshFilter>();
    }

    public void InitializePhysicsState(List<Vector3> customWorldPositions = null, List<Vector3> customRestPositions = null)
    {
        EnsureComponentsExist();
        cachedVisualShell = GetComponent<FEMVisualShell>();

        if (tetMesh == null || tetMesh.vertices == null) return;

        int n = tetMesh.vertices.Count;
        positions = new Vector3[n];
        restPositions = new Vector3[n];
        prevPositions = new Vector3[n];
        velocities = new Vector3[n];
        invMasses = new float[n];
        localVertsCache = new Vector3[n];

        for (int i = 0; i < n; i++)
        {
            positions[i] = (customWorldPositions != null && i < customWorldPositions.Count)
                ? customWorldPositions[i]
                : transform.TransformPoint(tetMesh.vertices[i]);

            restPositions[i] = (customRestPositions != null && i < customRestPositions.Count)
                ? customRestPositions[i]
                : transform.TransformPoint(tetMesh.vertices[i]);

            prevPositions[i] = positions[i];
            velocities[i] = Vector3.zero;
            invMasses[i] = 1.0f / Mathf.Max(0.0001f, particleMass);
        }

        PrecomputeRestMatrices();
        InitializeGPUResources();

        cachedSurfaceTriangles = tetMesh.ReconstructSurfaceTriangles().ToArray();

        MeshFilter mf = GetComponent<MeshFilter>();
        if (cachedVisualShell == null && mf != null)
        {
            if (mf.sharedMesh == null || mf.sharedMesh == sourceMesh || !mf.sharedMesh.name.Contains("FEM_VisualMesh"))
            {
                Mesh visualMesh = new Mesh { name = "FEM_VisualMesh" };
                visualMesh.MarkDynamic();
                mf.mesh = visualMesh;
            }
            UpdateVisualMesh(true);
        }

        UpdateColliderMesh();
        isInitialized = true;
    }

    private void PrecomputeRestMatrices()
    {
        if (tetMesh == null || tetMesh.tets == null) return;

        invRestMatrices = new GPUMatrix3x3[tetMesh.tets.Count];

        for (int t = 0; t < tetMesh.tets.Count; t++)
        {
            var tet = tetMesh.tets[t];

            if (!tet.active ||
                tet.v0 >= tetMesh.vertices.Count || tet.v1 >= tetMesh.vertices.Count ||
                tet.v2 >= tetMesh.vertices.Count || tet.v3 >= tetMesh.vertices.Count)
            {
                tet.active = false;
                tetMesh.tets[t] = tet;
                invRestMatrices[t] = new GPUMatrix3x3 { m00 = 1f, m11 = 1f, m22 = 1f };
                continue;
            }

            Vector3 X0 = transform.TransformPoint(tetMesh.vertices[tet.v0]);
            Vector3 X1 = transform.TransformPoint(tetMesh.vertices[tet.v1]);
            Vector3 X2 = transform.TransformPoint(tetMesh.vertices[tet.v2]);
            Vector3 X3 = transform.TransformPoint(tetMesh.vertices[tet.v3]);

            Vector3 Dm0 = X1 - X0;
            Vector3 Dm1 = X2 - X0;
            Vector3 Dm2 = X3 - X0;

            Vector3 c0 = Vector3.Cross(Dm1, Dm2);
            Vector3 c1 = Vector3.Cross(Dm2, Dm0);
            Vector3 c2 = Vector3.Cross(Dm0, Dm1);

            float det = Vector3.Dot(Dm0, c0);

            if (det < 0)
            {
                int tempV = tet.v1;
                tet.v1 = tet.v2;
                tet.v2 = tempV;

                X1 = transform.TransformPoint(tetMesh.vertices[tet.v1]);
                X2 = transform.TransformPoint(tetMesh.vertices[tet.v2]);

                Dm0 = X1 - X0;
                Dm1 = X2 - X0;

                c0 = Vector3.Cross(Dm1, Dm2);
                c1 = Vector3.Cross(Dm2, Dm0);
                c2 = Vector3.Cross(Dm0, Dm1);

                det = Vector3.Dot(Dm0, c0);
            }

            if (det < 1e-4f || float.IsNaN(det) || float.IsInfinity(det))
            {
                tet.active = false;
                tetMesh.tets[t] = tet;
                invRestMatrices[t] = new GPUMatrix3x3 { m00 = 1f, m11 = 1f, m22 = 1f };
                continue;
            }

            tet.restVolume = det / 6.0f;
            tetMesh.tets[t] = tet;

            float invDet = 1.0f / det;
            invRestMatrices[t] = new GPUMatrix3x3
            {
                m00 = c0.x * invDet,
                m01 = c1.x * invDet,
                m02 = c2.x * invDet,
                m10 = c0.y * invDet,
                m11 = c1.y * invDet,
                m12 = c2.y * invDet,
                m20 = c0.z * invDet,
                m21 = c1.z * invDet,
                m22 = c2.z * invDet
            };
        }
    }

    private void InitializeGPUResources()
    {
        ReleaseGPUResources();

        if (femComputeShader == null)
        {
            Debug.LogError("FEMPhysicsBody: Compute Shader reference is missing!");
            return;
        }

        int numVerts = (positions != null) ? positions.Length : 0;
        int numTets = (tetMesh != null && tetMesh.tets != null) ? tetMesh.tets.Count : 0;

        if (numVerts == 0 || numTets == 0) return;

        kIntegrate = femComputeShader.FindKernel("KernelIntegrate");
        kSolveFEM = femComputeShader.FindKernel("KernelSolveFEM");
        kApplyDeltas = femComputeShader.FindKernel("KernelApplyDeltas");
        kPostPhysics = femComputeShader.FindKernel("KernelPostPhysics");
        kParticleCollisions = femComputeShader.FindKernel("KernelParticleCollisions");
        kInterBodyCollisions = femComputeShader.FindKernel("KernelInterBodyCollisions");

        positionsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);
        prevPositionsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);
        velocitiesBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);
        invMassesBuffer = new ComputeBuffer(numVerts, sizeof(float));

        GPUTetrahedron[] gpuTets = new GPUTetrahedron[numTets];
        for (int i = 0; i < numTets; i++)
        {
            var tet = tetMesh.tets[i];
            gpuTets[i] = new GPUTetrahedron
            {
                v0 = tet.v0,
                v1 = tet.v1,
                v2 = tet.v2,
                v3 = tet.v3,
                restVolume = tet.restVolume,
                active = tet.active ? 1 : 0
            };
        }

        tetsBuffer = new ComputeBuffer(numTets, Marshal.SizeOf<GPUTetrahedron>());
        invRestMatricesBuffer = new ComputeBuffer(numTets, Marshal.SizeOf<GPUMatrix3x3>());
        deltaPosIntBuffer = new ComputeBuffer(numVerts, sizeof(int) * 3);
        deltaCountBuffer = new ComputeBuffer(numVerts, sizeof(int));

        positionsBuffer.SetData(positions);
        prevPositionsBuffer.SetData(prevPositions);
        velocitiesBuffer.SetData(velocities);
        invMassesBuffer.SetData(invMasses);
        tetsBuffer.SetData(gpuTets);
        invRestMatricesBuffer.SetData(invRestMatrices);

        BindBuffersToKernels();
    }

    private void BindBuffersToKernels()
    {
        int[] kernels = { kIntegrate, kSolveFEM, kApplyDeltas, kPostPhysics, kParticleCollisions, kInterBodyCollisions };
        foreach (int k in kernels)
        {
            femComputeShader.SetBuffer(k, "positionsBuffer", positionsBuffer);
            femComputeShader.SetBuffer(k, "prevPositionsBuffer", prevPositionsBuffer);
            femComputeShader.SetBuffer(k, "velocitiesBuffer", velocitiesBuffer);
            femComputeShader.SetBuffer(k, "invMassesBuffer", invMassesBuffer);
            femComputeShader.SetBuffer(k, "tetsBuffer", tetsBuffer);
            femComputeShader.SetBuffer(k, "invRestMatricesBuffer", invRestMatricesBuffer);
            femComputeShader.SetBuffer(k, "deltaPosIntBuffer", deltaPosIntBuffer);
            femComputeShader.SetBuffer(k, "deltaCountBuffer", deltaCountBuffer);
        }
    }

    private void FixedUpdate()
    {
        if (femComputeShader == null || positionsBuffer == null || !positionsBuffer.IsValid()) return;

        int numVerts = (positions != null) ? positions.Length : 0;
        int numTets = (tetMesh != null && tetMesh.tets != null) ? tetMesh.tets.Count : 0;

        if (numVerts == 0 || numTets == 0) return;

        int vertGroups = Mathf.Max(1, Mathf.CeilToInt(numVerts / 64.0f));
        int tetGroups = Mathf.Max(1, Mathf.CeilToInt(numTets / 64.0f));

        float dt = Time.fixedDeltaTime / solverSubsteps;
        float mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
        float lambda = (youngsModulus * poissonsRatio) / ((1.0f + poissonsRatio) * Mathf.Max(0.01f, 1.0f - 2.0f * poissonsRatio));
        float maxDisp = Mathf.Min(0.01f, targetTetSize * 0.2f);

        femComputeShader.SetInt("numVertices", numVerts);
        femComputeShader.SetInt("numTets", numTets);
        femComputeShader.SetFloat("dt", dt);
        femComputeShader.SetFloat("mu", mu);
        femComputeShader.SetFloat("lambda", lambda);
        femComputeShader.SetFloat("damping", damping);
        femComputeShader.SetVector("gravity", gravity);
        femComputeShader.SetFloat("maxDisplacement", maxDisp);
        femComputeShader.SetFloat("floorY", floorY);
        femComputeShader.SetInt("enableFloorCollision", enableFloorCollision ? 1 : 0);
        femComputeShader.SetFloat("particleRadius", particleRadius);

        for (int step = 0; step < solverSubsteps; step++)
        {
            femComputeShader.Dispatch(kIntegrate, vertGroups, 1, 1);

            if (enableSelfCollision)
            {
                femComputeShader.Dispatch(kParticleCollisions, vertGroups, 1, 1);
            }

            // --- INTER-BODY COLLISIONS ---
            for (int b = 0; b < allActiveBodies.Count; b++)
            {
                FEMPhysicsBody otherBody = allActiveBodies[b];
                if (otherBody == null || otherBody == this) continue;

                ComputeBuffer otherBuffer = otherBody.PositionsBuffer;
                if (otherBuffer != null && otherBuffer.IsValid() && otherBody.CurrentPositions != null && otherBody.CurrentPositions.Length > 0)
                {
                    femComputeShader.SetBuffer(kInterBodyCollisions, "positionsBuffer", positionsBuffer);
                    femComputeShader.SetBuffer(kInterBodyCollisions, "invMassesBuffer", invMassesBuffer);
                    femComputeShader.SetBuffer(kInterBodyCollisions, "deltaPosIntBuffer", deltaPosIntBuffer);
                    femComputeShader.SetBuffer(kInterBodyCollisions, "deltaCountBuffer", deltaCountBuffer);
                    femComputeShader.SetBuffer(kInterBodyCollisions, "otherPositionsBuffer", otherBuffer);

                    femComputeShader.SetInt("numVertices", numVerts);
                    femComputeShader.SetInt("numOtherVertices", otherBody.CurrentPositions.Length);
                    femComputeShader.SetFloat("particleRadius", particleRadius);

                    femComputeShader.Dispatch(kInterBodyCollisions, vertGroups, 1, 1);
                }
            }

            for (int iter = 0; iter < solverIterations; iter++)
            {
                femComputeShader.Dispatch(kSolveFEM, tetGroups, 1, 1);
                femComputeShader.Dispatch(kApplyDeltas, vertGroups, 1, 1);
            }

            femComputeShader.Dispatch(kPostPhysics, vertGroups, 1, 1);
        }

        positionsBuffer.GetData(positions);

        if (cachedVisualShell == null)
        {
            UpdateVisualMesh(false);
        }
    }

    private void UpdateLocalPositionsCache()
    {
        if (localVertsCache == null || localVertsCache.Length != positions.Length)
        {
            localVertsCache = new Vector3[positions.Length];
        }

        for (int i = 0; i < positions.Length; i++)
        {
            localVertsCache[i] = transform.InverseTransformPoint(positions[i]);
        }
    }

    private void UpdateVisualMesh(bool fullRebuild = false)
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        if (mf == null || tetMesh == null || mf.mesh == null) return;

        UpdateLocalPositionsCache();

        Mesh vMesh = mf.mesh;
        if (fullRebuild || cachedSurfaceTriangles == null || vMesh.vertexCount != localVertsCache.Length)
        {
            cachedSurfaceTriangles = tetMesh.ReconstructSurfaceTriangles().ToArray();
            vMesh.Clear();
            vMesh.vertices = localVertsCache;
            vMesh.triangles = cachedSurfaceTriangles;
        }
        else
        {
            vMesh.vertices = localVertsCache;
        }

        vMesh.RecalculateNormals();
        vMesh.RecalculateBounds();
    }

    private void UpdateColliderMesh()
    {
        if (positions == null || positions.Length == 0) return;

        BoxCollider boxCol = GetComponent<BoxCollider>();
        if (boxCol == null)
        {
            boxCol = gameObject.AddComponent<BoxCollider>();
        }

        Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        int activeCount = 0;

        if (tetMesh != null && tetMesh.tets != null)
        {
            bool[] activeVerts = new bool[positions.Length];
            foreach (var tet in tetMesh.tets)
            {
                if (!tet.active) continue;
                activeVerts[tet.v0] = true;
                activeVerts[tet.v1] = true;
                activeVerts[tet.v2] = true;
                activeVerts[tet.v3] = true;
            }

            for (int i = 0; i < positions.Length; i++)
            {
                if (activeVerts[i])
                {
                    Vector3 localPos = transform.InverseTransformPoint(positions[i]);
                    min = Vector3.Min(min, localPos);
                    max = Vector3.Max(max, localPos);
                    activeCount++;
                }
            }
        }
        else
        {
            for (int i = 0; i < positions.Length; i++)
            {
                Vector3 localPos = transform.InverseTransformPoint(positions[i]);
                min = Vector3.Min(min, localPos);
                max = Vector3.Max(max, localPos);
                activeCount++;
            }
        }

        if (activeCount == 0) return;

        Vector3 size = max - min;
        size.x = Mathf.Max(size.x, 0.01f);
        size.y = Mathf.Max(size.y, 0.01f);
        size.z = Mathf.Max(size.z, 0.01f);

        boxCol.center = (min + max) * 0.5f;
        boxCol.size = size;
    }

    private void ReleaseGPUResources()
    {
        if (positionsBuffer != null) { positionsBuffer.Release(); positionsBuffer = null; }
        if (prevPositionsBuffer != null) { prevPositionsBuffer.Release(); prevPositionsBuffer = null; }
        if (velocitiesBuffer != null) { velocitiesBuffer.Release(); velocitiesBuffer = null; }
        if (invMassesBuffer != null) { invMassesBuffer.Release(); invMassesBuffer = null; }
        if (tetsBuffer != null) { tetsBuffer.Release(); tetsBuffer = null; }
        if (invRestMatricesBuffer != null) { invRestMatricesBuffer.Release(); invRestMatricesBuffer = null; }
        if (deltaPosIntBuffer != null) { deltaPosIntBuffer.Release(); deltaPosIntBuffer = null; }
        if (deltaCountBuffer != null) { deltaCountBuffer.Release(); deltaCountBuffer = null; }
    }

    private void OnDestroy()
    {
        ReleaseGPUResources();
        if (generatedColliderMesh != null)
        {
            Destroy(generatedColliderMesh);
        }
    }

    private void OnDrawGizmos()
    {
        if (!drawTetWireframe || tetMesh == null || positions == null) return;

        Gizmos.color = Color.cyan;
        foreach (var tet in tetMesh.tets)
        {
            if (!tet.active) continue;
            if (tet.v0 >= positions.Length || tet.v1 >= positions.Length || tet.v2 >= positions.Length || tet.v3 >= positions.Length) continue;

            Gizmos.DrawLine(positions[tet.v0], positions[tet.v1]);
            Gizmos.DrawLine(positions[tet.v0], positions[tet.v2]);
            Gizmos.DrawLine(positions[tet.v0], positions[tet.v3]);
            Gizmos.DrawLine(positions[tet.v1], positions[tet.v2]);
            Gizmos.DrawLine(positions[tet.v1], positions[tet.v3]);
            Gizmos.DrawLine(positions[tet.v2], positions[tet.v3]);
        }
    }

    // ==========================================
    // CUTTING & CHUNK SEPARATION LOGIC
    // ==========================================

    public void OnMeshCut() => NotifyMeshCut();

    public void NotifyMeshCut()
    {
        if (tetMesh == null) return;

        MeshCutter.RecordCutForBody(this);

        CleanupDanglingTetrahedra();
        UpdateOrphanVertices();
        List<List<int>> islands = FindConnectedIslands();

        if (islands == null || islands.Count == 0)
        {
            Destroy(gameObject);
            return;
        }

        List<TetrahedralMesh> extractedMeshes = new List<TetrahedralMesh>();
        List<List<Vector3>> extractedPositions = new List<List<Vector3>>();
        List<List<Vector3>> extractedRestPositions = new List<List<Vector3>>();

        for (int i = 0; i < islands.Count; i++)
        {
            TetrahedralMesh subMesh = CreateSubTetMesh(islands[i], out List<Vector3> pos, out List<Vector3> restPos);
            extractedMeshes.Add(subMesh);
            extractedPositions.Add(pos);
            extractedRestPositions.Add(restPos);
        }

        if (extractedMeshes[0].tets.Count > 0 && extractedPositions[0].Count > 0)
        {
            this.tetMesh = extractedMeshes[0];
            InitializePhysicsState(extractedPositions[0], extractedRestPositions[0]);
        }
        else
        {
            Destroy(gameObject);
            return;
        }

        for (int i = 1; i < islands.Count; i++)
        {
            if (extractedMeshes[i].tets.Count > 0 && extractedPositions[i].Count > 0)
            {
                SpawnChunkGameObject(extractedMeshes[i], extractedPositions[i], extractedRestPositions[i]);
            }
        }

        FEMVisualShell[] shells = GetComponentsInChildren<FEMVisualShell>();
        foreach (var shell in shells)
        {
            shell.OnMeshCut();
        }
    }

    public void UpdateOrphanVertices()
    {
        if (positions == null || tetMesh == null || tetMesh.tets == null) return;

        bool[] activeVerts = new bool[positions.Length];

        foreach (var tet in tetMesh.tets)
        {
            if (!tet.active) continue;
            activeVerts[tet.v0] = true;
            activeVerts[tet.v1] = true;
            activeVerts[tet.v2] = true;
            activeVerts[tet.v3] = true;
        }

        for (int i = 0; i < positions.Length; i++)
        {
            if (!activeVerts[i])
            {
                velocities[i] = Vector3.zero;
                if (restPositions != null && i < restPositions.Length)
                    positions[i] = restPositions[i];
            }
        }
        UpdateColliderMesh();
    }

    private List<List<int>> FindConnectedIslands()
    {
        List<List<int>> islands = new List<List<int>>();
        if (tetMesh == null || tetMesh.tets == null) return islands;

        bool[] visited = new bool[tetMesh.tets.Count];

        Dictionary<int, List<int>> vertToTets = new Dictionary<int, List<int>>();
        for (int i = 0; i < tetMesh.tets.Count; i++)
        {
            var tet = tetMesh.tets[i];
            if (!tet.active) continue;

            int[] vList = { tet.v0, tet.v1, tet.v2, tet.v3 };
            foreach (int v in vList)
            {
                if (!vertToTets.TryGetValue(v, out var list))
                {
                    list = new List<int>();
                    vertToTets[v] = list;
                }
                list.Add(i);
            }
        }

        for (int i = 0; i < tetMesh.tets.Count; i++)
        {
            if (visited[i] || !tetMesh.tets[i].active) continue;

            List<int> currentIsland = new List<int>();
            Queue<int> queue = new Queue<int>();

            queue.Enqueue(i);
            visited[i] = true;

            while (queue.Count > 0)
            {
                int curr = queue.Dequeue();
                currentIsland.Add(curr);

                var tet = tetMesh.tets[curr];
                int[] vList = { tet.v0, tet.v1, tet.v2, tet.v3 };

                foreach (int v in vList)
                {
                    if (vertToTets.TryGetValue(v, out var neighbors))
                    {
                        foreach (int n in neighbors)
                        {
                            if (!visited[n] && tetMesh.tets[n].active)
                            {
                                visited[n] = true;
                                queue.Enqueue(n);
                            }
                        }
                    }
                }
            }
            islands.Add(currentIsland);
        }

        return islands;
    }

    public TetrahedralMesh CreateSubTetMesh(List<int> tetIndices, out List<Vector3> outPositions, out List<Vector3> outRestPositions)
    {
        TetrahedralMesh subMesh = new TetrahedralMesh
        {
            vertices = new List<Vector3>(),
            tets = new List<Tetrahedron>()
        };
        outPositions = new List<Vector3>();
        outRestPositions = new List<Vector3>();

        if (tetMesh == null || tetMesh.tets == null || tetIndices == null)
            return subMesh;

        Dictionary<int, int> oldToNewVertMap = new Dictionary<int, int>();

        foreach (int tetIdx in tetIndices)
        {
            if (tetIdx < 0 || tetIdx >= tetMesh.tets.Count) continue;

            var origTet = tetMesh.tets[tetIdx];
            if (!origTet.active) continue;

            int[] oldVerts = { origTet.v0, origTet.v1, origTet.v2, origTet.v3 };
            int[] newVerts = new int[4];

            for (int i = 0; i < 4; i++)
            {
                int oldV = oldVerts[i];
                if (!oldToNewVertMap.TryGetValue(oldV, out int newV))
                {
                    newV = subMesh.vertices.Count;
                    oldToNewVertMap[oldV] = newV;

                    if (oldV < tetMesh.vertices.Count)
                        subMesh.vertices.Add(tetMesh.vertices[oldV]);
                    else
                        subMesh.vertices.Add(Vector3.zero);

                    if (positions != null && oldV < positions.Length)
                        outPositions.Add(positions[oldV]);
                    else
                        outPositions.Add(transform.TransformPoint(subMesh.vertices[newV]));

                    if (restPositions != null && oldV < restPositions.Length)
                        outRestPositions.Add(restPositions[oldV]);
                    else
                        outRestPositions.Add(transform.TransformPoint(subMesh.vertices[newV]));
                }
                newVerts[i] = newV;
            }

            Tetrahedron newTet = new Tetrahedron
            {
                v0 = newVerts[0],
                v1 = newVerts[1],
                v2 = newVerts[2],
                v3 = newVerts[3],
                restVolume = origTet.restVolume,
                active = true
            };

            subMesh.tets.Add(newTet);
        }

        return subMesh;
    }

    private void SpawnChunkGameObject(TetrahedralMesh chunkMesh, List<Vector3> chunkPositions, List<Vector3> chunkRestPositions)
    {
        GameObject chunkGO = new GameObject(gameObject.name + "_Piece");
        chunkGO.transform.position = transform.position;
        chunkGO.transform.rotation = transform.rotation;
        chunkGO.transform.localScale = transform.localScale;

        MeshFilter mf = chunkGO.AddComponent<MeshFilter>();
        MeshRenderer mr = chunkGO.AddComponent<MeshRenderer>();

        MeshRenderer myRenderer = GetComponent<MeshRenderer>();
        if (myRenderer != null && myRenderer.sharedMaterials.Length > 0)
        {
            mr.sharedMaterials = myRenderer.sharedMaterials;
        }

        FEMPhysicsBody newBody = chunkGO.AddComponent<FEMPhysicsBody>();
        newBody.drawTetWireframe = false;
        newBody.sourceMesh = this.sourceMesh;
        newBody.femComputeShader = this.femComputeShader;
        newBody.youngsModulus = this.youngsModulus;
        newBody.poissonsRatio = this.poissonsRatio;
        newBody.solverSubsteps = this.solverSubsteps;
        newBody.solverIterations = this.solverIterations;
        newBody.particleMass = this.particleMass;
        newBody.damping = this.damping;
        newBody.gravity = this.gravity;
        newBody.enableFloorCollision = this.enableFloorCollision;
        newBody.floorY = this.floorY;

        BoxCollider mc = chunkGO.AddComponent<BoxCollider>();

        newBody.tetMesh = chunkMesh;
        newBody.InitializePhysicsState(chunkPositions, chunkRestPositions);

        MeshCutter.RecordCutForBody(newBody);
    }

    public void CleanupDanglingTetrahedra()
    {
        if (tetMesh == null || tetMesh.tets == null) return;
        Dictionary<int, int> vertTetCounts = new Dictionary<int, int>();
        for (int i = 0; i < tetMesh.tets.Count; i++)
        {
            var tet = tetMesh.tets[i];
            if (!tet.active) continue;

            vertTetCounts[tet.v0] = vertTetCounts.TryGetValue(tet.v0, out int c0) ? c0 + 1 : 1;
            vertTetCounts[tet.v1] = vertTetCounts.TryGetValue(tet.v1, out int c1) ? c1 + 1 : 1;
            vertTetCounts[tet.v2] = vertTetCounts.TryGetValue(tet.v2, out int c2) ? c2 + 1 : 1;
            vertTetCounts[tet.v3] = vertTetCounts.TryGetValue(tet.v3, out int c3) ? c3 + 1 : 1;
        }

        for (int i = 0; i < tetMesh.tets.Count; i++)
        {
            var tet = tetMesh.tets[i];
            if (!tet.active) continue;

            if (vertTetCounts[tet.v0] == 1 && vertTetCounts[tet.v1] == 1 &&
                vertTetCounts[tet.v2] == 1 && vertTetCounts[tet.v3] == 1)
            {
                tet.active = false;
                tetMesh.tets[i] = tet;
            }
        }
    }
}