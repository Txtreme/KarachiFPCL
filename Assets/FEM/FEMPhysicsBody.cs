using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class FEMPhysicsBody : MonoBehaviour
{
    [Header("Source Mesh Configuration")]
    [Tooltip("Assign the source 3D volume mesh (e.g. Sphere, Cube, FBX). If left empty, auto-detects from MeshFilter on Awake.")]
    public Mesh sourceMesh;

    [Header("Compute Shader Reference")]
    public ComputeShader femComputeShader;

    [Header("Tetrahedrization & Element Size")]
    public float targetTetSize = 0.05f;
    public Vector3Int gridResolution = new Vector3Int(8, 8, 8);

    [Header("FEM Physical Parameters")]
    [Range(1, 30)] public int solverSubsteps = 10;
    public float particleMass = 0.1f;
    public Vector3 gravity = new Vector3(0, -9.81f, 0);
    [Range(0f, 1f)] public float damping = 0.02f;

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

    private GPUMatrix3x3[] invRestMatrices;

    // GPU Compute Buffers
    private ComputeBuffer positionsBuffer;
    private ComputeBuffer prevPositionsBuffer;
    private ComputeBuffer velocitiesBuffer;
    private ComputeBuffer invMassesBuffer;
    private ComputeBuffer tetsBuffer;
    private ComputeBuffer invRestMatricesBuffer;
    private ComputeBuffer deltaPosIntBuffer;
    private ComputeBuffer deltaCountBuffer;

    private int kIntegrate, kSolveFEM, kApplyDeltas, kPostPhysics;

    public struct GPUTetrahedron
    {
        public int v0, v1, v2, v3;
        public float restVolume;
        public int active;
    }

    [System.Serializable]
    public struct GPUMatrix3x3
    {
        public float m00, m01, m02;
        public float m10, m11, m12;
        public float m20, m21, m22;
    }

    public TetrahedralMesh TetMesh => tetMesh;
    public Vector3[] CurrentPositions => positions;

    private void Awake()
    {
        EnsureComponentsExist();

        if (sourceMesh == null)
        {
            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf != null && mf.sharedMesh != null)
            {
                if (!mf.sharedMesh.name.Contains("FEM_Visual_Shell"))
                {
                    sourceMesh = mf.sharedMesh;
                }
            }
        }
    }

    private void Start()
    {
        if (tetMesh == null)
        {
            if (sourceMesh == null)
            {
                Debug.LogError("FEMPhysicsBody: No source Mesh assigned or found on MeshFilter!");
                return;
            }

            if (!sourceMesh.isReadable)
            {
                Debug.LogError($"FEMPhysicsBody: Mesh '{sourceMesh.name}' is not Read/Write enabled! Enable 'Read/Write' in model import settings.");
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
            Debug.LogError("FEMPhysicsBody: Tetrahedral generation returned 0 elements. Try lowering 'targetTetSize' or increasing 'gridResolution'.");
            return;
        }

        InitializePhysicsState();
    }

    private void EnsureComponentsExist()
    {
        if (GetComponent<MeshFilter>() == null) gameObject.AddComponent<MeshFilter>();
    }

    public void InitializePhysicsState(List<Vector3> customWorldPositions = null)
    {
        EnsureComponentsExist();

        int n = tetMesh.vertices.Count;
        positions = new Vector3[n];
        prevPositions = new Vector3[n];
        velocities = new Vector3[n];
        invMasses = new float[n];

        for (int i = 0; i < n; i++)
        {
            positions[i] = (customWorldPositions != null && i < customWorldPositions.Count)
                ? customWorldPositions[i]
                : transform.TransformPoint(tetMesh.vertices[i]);

            prevPositions[i] = positions[i];
            velocities[i] = Vector3.zero;
            invMasses[i] = 1.0f / particleMass;
        }

        PrecomputeRestMatrices();
        InitializeGPUResources();

        if (GetComponent<FEMVisualShell>() == null)
        {
            Mesh visualMesh = new Mesh();
            visualMesh.MarkDynamic();
            GetComponent<MeshFilter>().mesh = visualMesh;
            UpdateVisualMesh();
        }

        UpdateColliderMesh();
    }

    private void PrecomputeRestMatrices()
    {
        invRestMatrices = new GPUMatrix3x3[tetMesh.tets.Count];

        for (int t = 0; t < tetMesh.tets.Count; t++)
        {
            var tet = tetMesh.tets[t];

            Vector3 X0 = tetMesh.vertices[tet.v0];
            Vector3 X1 = tetMesh.vertices[tet.v1];
            Vector3 X2 = tetMesh.vertices[tet.v2];
            Vector3 X3 = tetMesh.vertices[tet.v3];

            Vector3 Dm0 = X1 - X0;
            Vector3 Dm1 = X2 - X0;
            Vector3 Dm2 = X3 - X0;

            FEMMatrix3x3 Dm = new FEMMatrix3x3(Dm0, Dm1, Dm2);
            float det = Dm.Determinant();

            if (det < 0)
            {
                int temp = tet.v1;
                tet.v1 = tet.v2;
                tet.v2 = temp;

                Dm0 = tetMesh.vertices[tet.v1] - X0;
                Dm1 = tetMesh.vertices[tet.v2] - X0;
                Dm = new FEMMatrix3x3(Dm0, Dm1, Dm2);
                det = Dm.Determinant();
            }

            tet.restVolume = Mathf.Max(det / 6.0f, 1e-6f);
            tetMesh.tets[t] = tet;

            FEMMatrix3x3 invDm = (Mathf.Abs(det) > 1e-7f) ? Dm.Inverse() : FEMMatrix3x3.Identity;

            invRestMatrices[t] = new GPUMatrix3x3
            {
                m00 = invDm.m00,
                m01 = invDm.m01,
                m02 = invDm.m02,
                m10 = invDm.m10,
                m11 = invDm.m11,
                m12 = invDm.m12,
                m20 = invDm.m20,
                m21 = invDm.m21,
                m22 = invDm.m22
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

        if (numVerts == 0 || numTets == 0)
        {
            Debug.LogError($"FEMPhysicsBody: Cannot initialize GPU resources. Vertices: {numVerts}, Tetrahedra: {numTets}.");
            return;
        }

        kIntegrate = femComputeShader.FindKernel("KernelIntegrate");
        kSolveFEM = femComputeShader.FindKernel("KernelSolveFEM");
        kApplyDeltas = femComputeShader.FindKernel("KernelApplyDeltas");
        kPostPhysics = femComputeShader.FindKernel("KernelPostPhysics");

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
        int[] kernels = { kIntegrate, kSolveFEM, kApplyDeltas, kPostPhysics };
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
        if (femComputeShader == null || positionsBuffer == null || !positionsBuffer.IsValid() || tetsBuffer == null || !tetsBuffer.IsValid()) return;

        int numVerts = (positions != null) ? positions.Length : 0;
        int numTets = (tetMesh != null && tetMesh.tets != null) ? tetMesh.tets.Count : 0;

        if (numVerts == 0 || numTets == 0) return;

        int vertGroups = Mathf.Max(1, Mathf.CeilToInt(numVerts / 64.0f));
        int tetGroups = Mathf.Max(1, Mathf.CeilToInt(numTets / 64.0f));

        float dt = Time.fixedDeltaTime / solverSubsteps;
        float mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
        float lambda = (youngsModulus * poissonsRatio) / ((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));

        femComputeShader.SetInt("numVertices", numVerts);
        femComputeShader.SetInt("numTets", numTets);
        femComputeShader.SetFloat("dt", dt);
        femComputeShader.SetFloat("mu", mu);
        femComputeShader.SetFloat("lambda", lambda);
        femComputeShader.SetFloat("damping", damping);
        femComputeShader.SetVector("gravity", gravity);
        femComputeShader.SetFloat("maxDisplacement", 0.05f);
        femComputeShader.SetFloat("floorY", floorY);
        femComputeShader.SetInt("enableFloorCollision", enableFloorCollision ? 1 : 0);

        for (int step = 0; step < solverSubsteps; step++)
        {
            femComputeShader.Dispatch(kIntegrate, vertGroups, 1, 1);
            femComputeShader.Dispatch(kSolveFEM, tetGroups, 1, 1);
            femComputeShader.Dispatch(kApplyDeltas, vertGroups, 1, 1);
            femComputeShader.Dispatch(kPostPhysics, vertGroups, 1, 1);
        }

        positionsBuffer.GetData(positions);

        if (GetComponent<FEMVisualShell>() == null)
        {
            UpdateVisualMesh();
        }
    }

    public void NotifyMeshCut()
    {
        CleanupDanglingTetrahedra();
        PrecomputeRestMatrices();
        InitializeGPUResources();

        if (GetComponent<FEMVisualShell>() == null)
        {
            UpdateVisualMesh();
        }

        UpdateColliderMesh();

        FEMVisualShell visualShell = GetComponent<FEMVisualShell>();
        if (visualShell != null)
        {
            visualShell.OnMeshCut();
        }
    }

    public void CleanupDanglingTetrahedra()
    {
        if (tetMesh == null || tetMesh.tets == null) return;

        int maxPasses = 3;
        for (int pass = 0; pass < maxPasses; pass++)
        {
            bool removedInPass = false;

            Dictionary<int, int> vertTetCounts = new Dictionary<int, int>();
            for (int i = 0; i < tetMesh.tets.Count; i++)
            {
                var tet = tetMesh.tets[i];
                if (!tet.active) continue;

                vertTetCounts[tet.v0] = vertTetCounts.GetValueOrDefault(tet.v0, 0) + 1;
                vertTetCounts[tet.v1] = vertTetCounts.GetValueOrDefault(tet.v1, 0) + 1;
                vertTetCounts[tet.v2] = vertTetCounts.GetValueOrDefault(tet.v2, 0) + 1;
                vertTetCounts[tet.v3] = vertTetCounts.GetValueOrDefault(tet.v3, 0) + 1;
            }

            for (int i = 0; i < tetMesh.tets.Count; i++)
            {
                var tet = tetMesh.tets[i];
                if (!tet.active) continue;

                int lonelyVerts = 0;
                if (vertTetCounts.GetValueOrDefault(tet.v0, 0) == 1) lonelyVerts++;
                if (vertTetCounts.GetValueOrDefault(tet.v1, 0) == 1) lonelyVerts++;
                if (vertTetCounts.GetValueOrDefault(tet.v2, 0) == 1) lonelyVerts++;
                if (vertTetCounts.GetValueOrDefault(tet.v3, 0) == 1) lonelyVerts++;

                if (lonelyVerts >= 2)
                {
                    tet.active = false;
                    tetMesh.tets[i] = tet;
                    removedInPass = true;
                }
            }

            if (!removedInPass) break;
        }
    }

    private void UpdateVisualMesh()
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        if (mf == null || tetMesh == null || mf.mesh == null) return;

        List<int> tris = tetMesh.ReconstructSurfaceTriangles();

        Vector3[] localVerts = new Vector3[positions.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            localVerts[i] = transform.InverseTransformPoint(positions[i]);
        }

        Mesh vMesh = mf.mesh;
        vMesh.Clear();
        vMesh.vertices = localVerts;
        vMesh.triangles = tris.ToArray();
        vMesh.RecalculateNormals();
        vMesh.RecalculateBounds();
    }

    private void UpdateColliderMesh()
    {
        MeshCollider mc = GetComponent<MeshCollider>();
        if (mc == null) return;

        List<int> rawTris = tetMesh.ReconstructSurfaceTriangles();
        List<int> validTris = GetNonDegenerateTriangles(rawTris);

        if (validTris.Count < 3)
        {
            mc.sharedMesh = null;
            return;
        }

        Mesh colMesh = new Mesh();
        Vector3[] localVerts = new Vector3[positions.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            localVerts[i] = transform.InverseTransformPoint(positions[i]);
        }

        colMesh.vertices = localVerts;
        colMesh.triangles = validTris.ToArray();
        colMesh.RecalculateBounds();

        mc.sharedMesh = null;
        mc.sharedMesh = colMesh;
    }

    private List<int> GetNonDegenerateTriangles(List<int> rawTris)
    {
        List<int> validTris = new List<int>();
        for (int i = 0; i < rawTris.Count; i += 3)
        {
            Vector3 v0 = positions[rawTris[i]];
            Vector3 v1 = positions[rawTris[i + 1]];
            Vector3 v2 = positions[rawTris[i + 2]];

            if (Vector3.Cross(v1 - v0, v2 - v0).sqrMagnitude > 1e-7f)
            {
                validTris.Add(rawTris[i]);
                validTris.Add(rawTris[i + 1]);
                validTris.Add(rawTris[i + 2]);
            }
        }
        return validTris;
    }

    private void ReleaseGPUResources()
    {
        positionsBuffer?.Release();
        prevPositionsBuffer?.Release();
        velocitiesBuffer?.Release();
        invMassesBuffer?.Release();
        tetsBuffer?.Release();
        invRestMatricesBuffer?.Release();
        deltaPosIntBuffer?.Release();
        deltaCountBuffer?.Release();
    }

    private void OnDisable() => ReleaseGPUResources();
    private void OnDestroy() => ReleaseGPUResources();

    private void OnDrawGizmos()
    {
        if (!drawTetWireframe || tetMesh == null || positions == null) return;

        Gizmos.color = Color.cyan;
        foreach (var tet in tetMesh.tets)
        {
            if (!tet.active) continue;
            Gizmos.DrawLine(positions[tet.v0], positions[tet.v1]);
            Gizmos.DrawLine(positions[tet.v0], positions[tet.v2]);
            Gizmos.DrawLine(positions[tet.v0], positions[tet.v3]);
            Gizmos.DrawLine(positions[tet.v1], positions[tet.v2]);
            Gizmos.DrawLine(positions[tet.v1], positions[tet.v3]);
            Gizmos.DrawLine(positions[tet.v2], positions[tet.v3]);
        }
    }
}