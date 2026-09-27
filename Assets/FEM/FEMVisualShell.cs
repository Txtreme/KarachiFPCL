using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class FEMVisualShell : MonoBehaviour
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct GPUTet
    {
        public int v0, v1, v2, v3;
    }

    public enum ShellMode
    {
        ProceduralSurface,
        CustomMeshBarycentric
    }

    [Header("Compute Shader Reference")]
    public ComputeShader skinningComputeShader;

    [Header("Shell Configuration")]
    public FEMPhysicsBody physicsBody;
    public ShellMode shellMode = ShellMode.CustomMeshBarycentric;

    [Header("Custom High-Poly Model")]
    public Mesh customVisualMesh;

    private Mesh shellMesh;
    private MeshFilter meshFilter;
    private bool isInitialized = false;

    // Source Data
    private Vector3[] customSourceVertices;
    private Vector3[] deformedVertices;
    private Vector3[] localVerts; // Cached to eliminate per-frame GC allocations
    private int[] customTetIndices;
    private Vector4[] customBaryWeights;

    // GPU Compute Buffers
    private ComputeBuffer tetBuffer;
    private ComputeBuffer physPosBuffer;
    private ComputeBuffer tetIdxBuffer;
    private ComputeBuffer baryWeightsBuffer;
    private ComputeBuffer outputVertsBuffer;

    private int kernelCSMain;

    private void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        if (physicsBody == null) physicsBody = GetComponentInParent<FEMPhysicsBody>();
    }

    private void Start()
    {
        RebuildShell();
    }

    private void LateUpdate()
    {
        if (!isInitialized || physicsBody == null) return;

        if (shellMode == ShellMode.CustomMeshBarycentric && skinningComputeShader != null && outputVertsBuffer != null && tetBuffer != null && tetBuffer.IsValid())
        {
            physPosBuffer.SetData(physicsBody.CurrentPositions);
            skinningComputeShader.SetBuffer(kernelCSMain, "_PhysPosBuffer", physPosBuffer);

            int threadGroups = Mathf.Max(1, Mathf.CeilToInt(customSourceVertices.Length / 64.0f));
            skinningComputeShader.Dispatch(kernelCSMain, threadGroups, 1, 1);

            outputVertsBuffer.GetData(deformedVertices);

            // Reallocate cached localVerts array only when vertex count changes
            if (localVerts == null || localVerts.Length != deformedVertices.Length)
            {
                localVerts = new Vector3[deformedVertices.Length];
            }

            Matrix4x4 worldToLocal = transform.worldToLocalMatrix;
            for (int i = 0; i < deformedVertices.Length; i++)
            {
                localVerts[i] = worldToLocal.MultiplyPoint3x4(deformedVertices[i]);
            }

            shellMesh.vertices = localVerts;
            shellMesh.RecalculateNormals();
            shellMesh.RecalculateBounds();
            meshFilter.mesh = shellMesh;
        }
        else if (shellMode == ShellMode.ProceduralSurface)
        {
            SetupProceduralSurfaceShell();
        }
    }

    public void OnMeshCut()
    {
        isInitialized = false;
        RebuildShell();
    }

    public void RebuildShell()
    {
        ReleaseComputeBuffers();

        if (physicsBody == null || physicsBody.TetMesh == null || physicsBody.CurrentPositions == null) return;

        if (shellMesh == null)
        {
            shellMesh = new Mesh { name = "FEM_GPU_Visual_Shell" };
            shellMesh.MarkDynamic();
            if (meshFilter != null) meshFilter.mesh = shellMesh;
        }

        if (shellMode == ShellMode.CustomMeshBarycentric && customVisualMesh != null && skinningComputeShader != null)
        {
            SetupBarycentricSkinningSpatial();
            InitializeGPUBuffers();
        }
        else
        {
            SetupProceduralSurfaceShell();
        }

        isInitialized = true;
    }

    private void SetupBarycentricSkinningSpatial()
    {
        var tetMesh = physicsBody.TetMesh;
        if (tetMesh == null || tetMesh.tets.Count == 0) return;

        customSourceVertices = customVisualMesh.vertices;
        int numVerts = customSourceVertices.Length;

        customTetIndices = new int[numVerts];
        customBaryWeights = new Vector4[numVerts];

        Vector3 currentScale = transform.lossyScale;
        float maxScaleFactor = Mathf.Max(currentScale.x, Mathf.Max(currentScale.y, currentScale.z));
        float cellSize = physicsBody.targetTetSize * maxScaleFactor * 2.0f;
        if (cellSize <= 0.001f) cellSize = 0.2f * maxScaleFactor;

        Dictionary<Vector3Int, List<int>> spatialGrid = new Dictionary<Vector3Int, List<int>>();

        Vector3[] scaledSourceVerts = new Vector3[numVerts];
        for (int i = 0; i < numVerts; i++)
        {
            scaledSourceVerts[i] = transform.TransformPoint(customSourceVertices[i]);
        }

        for (int t = 0; t < tetMesh.tets.Count; t++)
        {
            var tet = tetMesh.tets[t];
            if (!tet.active) continue;

            Vector3 center = (physicsBody.CurrentPositions[tet.v0] +
                              physicsBody.CurrentPositions[tet.v1] +
                              physicsBody.CurrentPositions[tet.v2] +
                              physicsBody.CurrentPositions[tet.v3]) * 0.25f;

            Vector3Int cell = GetGridCell(center, cellSize);
            if (!spatialGrid.TryGetValue(cell, out List<int> list))
            {
                list = new List<int>();
                spatialGrid[cell] = list;
            }
            list.Add(t);
        }

        for (int i = 0; i < numVerts; i++)
        {
            Vector3 worldVPos = scaledSourceVerts[i];
            Vector3Int cell = GetGridCell(worldVPos, cellSize);

            int bestTetIdx = -1;
            Vector4 bestWeights = Vector4.zero;
            float minDistance = float.MaxValue;

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        Vector3Int neighborCell = cell + new Vector3Int(x, y, z);
                        if (!spatialGrid.TryGetValue(neighborCell, out List<int> candidateTets)) continue;

                        foreach (int t in candidateTets)
                        {
                            var tet = tetMesh.tets[t];
                            Vector3 x0 = physicsBody.CurrentPositions[tet.v0];
                            Vector3 x1 = physicsBody.CurrentPositions[tet.v1];
                            Vector3 x2 = physicsBody.CurrentPositions[tet.v2];
                            Vector3 x3 = physicsBody.CurrentPositions[tet.v3];

                            Vector3 tetCenter = (x0 + x1 + x2 + x3) * 0.25f;
                            float dist = (worldVPos - tetCenter).sqrMagnitude;

                            if (dist < minDistance)
                            {
                                FEMMatrix3x3 Dm = new FEMMatrix3x3(x1 - x0, x2 - x0, x3 - x0);
                                FEMMatrix3x3 invDm = Dm.Inverse();

                                Vector3 diff = worldVPos - x0;
                                Vector3 abg = invDm.MultiplyVector(diff);

                                float w1 = abg.x;
                                float w2 = abg.y;
                                float w3 = abg.z;
                                float w0 = 1.0f - (w1 + w2 + w3);

                                minDistance = dist;
                                bestTetIdx = t;
                                bestWeights = new Vector4(w0, w1, w2, w3);
                            }
                        }
                    }
                }
            }

            // Global fallback search if local spatial grid lookup yielded no candidates
            if (bestTetIdx == -1 && tetMesh.tets.Count > 0)
            {
                for (int t = 0; t < tetMesh.tets.Count; t++)
                {
                    var tet = tetMesh.tets[t];
                    if (!tet.active) continue;

                    Vector3 x0 = physicsBody.CurrentPositions[tet.v0];
                    Vector3 x1 = physicsBody.CurrentPositions[tet.v1];
                    Vector3 x2 = physicsBody.CurrentPositions[tet.v2];
                    Vector3 x3 = physicsBody.CurrentPositions[tet.v3];

                    Vector3 tetCenter = (x0 + x1 + x2 + x3) * 0.25f;
                    float dist = (worldVPos - tetCenter).sqrMagnitude;

                    if (dist < minDistance)
                    {
                        FEMMatrix3x3 Dm = new FEMMatrix3x3(x1 - x0, x2 - x0, x3 - x0);
                        FEMMatrix3x3 invDm = Dm.Inverse();

                        Vector3 diff = worldVPos - x0;
                        Vector3 abg = invDm.MultiplyVector(diff);

                        float w1 = abg.x;
                        float w2 = abg.y;
                        float w3 = abg.z;
                        float w0 = 1.0f - (w1 + w2 + w3);

                        minDistance = dist;
                        bestTetIdx = t;
                        bestWeights = new Vector4(w0, w1, w2, w3);
                    }
                }
            }

            if (bestTetIdx == -1)
            {
                bestTetIdx = 0;
                bestWeights = new Vector4(0.25f, 0.25f, 0.25f, 0.25f);
            }

            customTetIndices[i] = bestTetIdx;
            customBaryWeights[i] = bestWeights;
        }

        shellMesh.Clear();
        shellMesh.vertices = new Vector3[numVerts];
        shellMesh.triangles = customVisualMesh.triangles;
        shellMesh.uv = customVisualMesh.uv;
    }

    private void SetupProceduralSurfaceShell()
    {
        if (physicsBody == null || physicsBody.TetMesh == null) return;

        List<int> surfaceTris = physicsBody.TetMesh.ReconstructSurfaceTriangles();
        Vector3[] localPositions = new Vector3[physicsBody.CurrentPositions.Length];

        for (int i = 0; i < physicsBody.CurrentPositions.Length; i++)
        {
            localPositions[i] = transform.InverseTransformPoint(physicsBody.CurrentPositions[i]);
        }

        shellMesh.Clear();
        shellMesh.vertices = localPositions;
        shellMesh.triangles = surfaceTris.ToArray();
        shellMesh.RecalculateNormals();
        shellMesh.RecalculateBounds();
        meshFilter.mesh = shellMesh;
    }

    private void InitializeGPUBuffers()
    {
        var tetMesh = physicsBody.TetMesh;
        if (tetMesh == null || customSourceVertices == null) return;

        int numVerts = customSourceVertices.Length;
        int numTets = tetMesh.tets.Count;

        deformedVertices = new Vector3[numVerts];

        GPUTet[] gpuTets = new GPUTet[numTets];
        for (int i = 0; i < numTets; i++)
        {
            var t = tetMesh.tets[i];
            gpuTets[i] = new GPUTet { v0 = t.v0, v1 = t.v1, v2 = t.v2, v3 = t.v3 };
        }

        tetBuffer = new ComputeBuffer(numTets, sizeof(int) * 4);
        tetBuffer.SetData(gpuTets);

        physPosBuffer = new ComputeBuffer(physicsBody.CurrentPositions.Length, sizeof(float) * 3);
        tetIdxBuffer = new ComputeBuffer(numVerts, sizeof(int));
        baryWeightsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 4);
        outputVertsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);

        tetIdxBuffer.SetData(customTetIndices);
        baryWeightsBuffer.SetData(customBaryWeights);

        kernelCSMain = skinningComputeShader.FindKernel("CSMain");

        skinningComputeShader.SetBuffer(kernelCSMain, "_TetBuffer", tetBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_PhysPosBuffer", physPosBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_CustomTetIndices", tetIdxBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_CustomBaryWeights", baryWeightsBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_DeformedVertices", outputVertsBuffer);
        skinningComputeShader.SetInt("numVertices", numVerts);
    }

    private Vector3Int GetGridCell(Vector3 pos, float cellSize)
    {
        return new Vector3Int(
            Mathf.FloorToInt(pos.x / cellSize),
            Mathf.FloorToInt(pos.y / cellSize),
            Mathf.FloorToInt(pos.z / cellSize)
        );
    }

    private void ReleaseComputeBuffers()
    {
        tetBuffer?.Release();
        physPosBuffer?.Release();
        tetIdxBuffer?.Release();
        baryWeightsBuffer?.Release();
        outputVertsBuffer?.Release();
    }

    private void OnDisable() => ReleaseComputeBuffers();
    private void OnDestroy() => ReleaseComputeBuffers();
}