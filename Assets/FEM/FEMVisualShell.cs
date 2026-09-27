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
    private MeshRenderer meshRenderer;
    private MaterialPropertyBlock mpb;
    private bool isInitialized = false;

    // Source Data
    private Vector3[] customSourceVertices;
    private Vector3[] customSourceNormals;
    private int[] customTetIndices;
    private Vector4[] customBaryWeights;

    // GPU Compute Buffers
    private ComputeBuffer tetBuffer;
    private ComputeBuffer physPosBuffer;
    private ComputeBuffer restPosBuffer;
    private ComputeBuffer tetIdxBuffer;
    private ComputeBuffer baryWeightsBuffer;
    private ComputeBuffer sourceNormalsBuffer;
    private ComputeBuffer outputVertsBuffer;
    private ComputeBuffer outputNormalsBuffer;

    private int kernelCSMain;

    private void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        mpb = new MaterialPropertyBlock();
        if (physicsBody == null) physicsBody = GetComponentInParent<FEMPhysicsBody>();
    }

    private void Start()
    {
        RebuildShell();
    }

    private void LateUpdate()
    {
        if (!isInitialized)
        {
            RebuildShell();
            if (!isInitialized) return;
        }

        if (shellMode == ShellMode.CustomMeshBarycentric && skinningComputeShader != null && outputVertsBuffer != null && tetBuffer != null && tetBuffer.IsValid())
        {
            var currentPos = physicsBody.CurrentPositions;
            if (currentPos == null || physPosBuffer == null || physPosBuffer.count != currentPos.Length)
            {
                RebuildShell();
                return;
            }

            physPosBuffer.SetData(currentPos);

            int threadGroups = Mathf.Max(1, Mathf.CeilToInt(customSourceVertices.Length / 64.0f));
            skinningComputeShader.Dispatch(kernelCSMain, threadGroups, 1, 1);

            meshRenderer.GetPropertyBlock(mpb);
            mpb.SetBuffer("_DeformedVertices", outputVertsBuffer);
            mpb.SetBuffer("_DeformedNormals", outputNormalsBuffer);
            meshRenderer.SetPropertyBlock(mpb);
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

        if (physicsBody == null || physicsBody.TetMesh == null || physicsBody.CurrentPositions == null || physicsBody.CurrentPositions.Length == 0)
        {
            isInitialized = false;
            return;
        }

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

        Vector3[] basePositions = physicsBody.RestPositions ?? physicsBody.CurrentPositions;

        customSourceVertices = customVisualMesh.vertices;
        int numVerts = customSourceVertices.Length;

        customSourceNormals = new Vector3[numVerts];
        Vector3[] rawNormals = customVisualMesh.normals;
        bool hasNormals = rawNormals != null && rawNormals.Length == numVerts;

        for (int i = 0; i < numVerts; i++)
        {
            Vector3 localN = hasNormals ? rawNormals[i] : Vector3.up;
            customSourceNormals[i] = transform.TransformDirection(localN).normalized;
        }

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

            Vector3 center = (basePositions[tet.v0] +
                              basePositions[tet.v1] +
                              basePositions[tet.v2] +
                              basePositions[tet.v3]) * 0.25f;

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
                            Vector3 x0 = basePositions[tet.v0];
                            Vector3 x1 = basePositions[tet.v1];
                            Vector3 x2 = basePositions[tet.v2];
                            Vector3 x3 = basePositions[tet.v3];

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

            if (bestTetIdx == -1 && tetMesh.tets.Count > 0)
            {
                for (int t = 0; t < tetMesh.tets.Count; t++)
                {
                    var tet = tetMesh.tets[t];
                    if (!tet.active) continue;

                    Vector3 x0 = basePositions[tet.v0];
                    Vector3 x1 = basePositions[tet.v1];
                    Vector3 x2 = basePositions[tet.v2];
                    Vector3 x3 = basePositions[tet.v3];

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
        shellMesh.vertices = customSourceVertices;
        shellMesh.triangles = customVisualMesh.triangles;
        shellMesh.uv = customVisualMesh.uv;
        shellMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 100000.0f);
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
        int numPhysNodes = physicsBody.CurrentPositions.Length;

        GPUTet[] gpuTets = new GPUTet[numTets];
        for (int i = 0; i < numTets; i++)
        {
            var t = tetMesh.tets[i];
            gpuTets[i] = new GPUTet { v0 = t.v0, v1 = t.v1, v2 = t.v2, v3 = t.v3 };
        }

        tetBuffer = new ComputeBuffer(numTets, sizeof(int) * 4);
        tetBuffer.SetData(gpuTets);

        physPosBuffer = new ComputeBuffer(numPhysNodes, sizeof(float) * 3);
        restPosBuffer = new ComputeBuffer(numPhysNodes, sizeof(float) * 3);
        tetIdxBuffer = new ComputeBuffer(numVerts, sizeof(int));
        baryWeightsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 4);
        sourceNormalsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);

        outputVertsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);
        outputNormalsBuffer = new ComputeBuffer(numVerts, sizeof(float) * 3);

        physPosBuffer.SetData(physicsBody.CurrentPositions);
        restPosBuffer.SetData(physicsBody.RestPositions ?? physicsBody.CurrentPositions);
        tetIdxBuffer.SetData(customTetIndices);
        baryWeightsBuffer.SetData(customBaryWeights);
        sourceNormalsBuffer.SetData(customSourceNormals);

        kernelCSMain = skinningComputeShader.FindKernel("CSMain");

        skinningComputeShader.SetBuffer(kernelCSMain, "_TetBuffer", tetBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_PhysPosBuffer", physPosBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_RestPosBuffer", restPosBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_CustomTetIndices", tetIdxBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_CustomBaryWeights", baryWeightsBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_CustomSourceNormals", sourceNormalsBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_DeformedVertices", outputVertsBuffer);
        skinningComputeShader.SetBuffer(kernelCSMain, "_DeformedNormals", outputNormalsBuffer);
        skinningComputeShader.SetInt("numVertices", numVerts);

        int threadGroups = Mathf.Max(1, Mathf.CeilToInt(numVerts / 64.0f));
        skinningComputeShader.Dispatch(kernelCSMain, threadGroups, 1, 1);

        if (meshRenderer != null)
        {
            meshRenderer.GetPropertyBlock(mpb);
            mpb.SetBuffer("_DeformedVertices", outputVertsBuffer);
            mpb.SetBuffer("_DeformedNormals", outputNormalsBuffer);
            meshRenderer.SetPropertyBlock(mpb);
        }
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
        tetBuffer?.Release(); tetBuffer = null;
        physPosBuffer?.Release(); physPosBuffer = null;
        restPosBuffer?.Release(); restPosBuffer = null;
        tetIdxBuffer?.Release(); tetIdxBuffer = null;
        baryWeightsBuffer?.Release(); baryWeightsBuffer = null;
        sourceNormalsBuffer?.Release(); sourceNormalsBuffer = null;
        outputVertsBuffer?.Release(); outputVertsBuffer = null;
        outputNormalsBuffer?.Release(); outputNormalsBuffer = null;
    }

    private void OnDisable() => ReleaseComputeBuffers();
    private void OnDestroy() => ReleaseComputeBuffers();
}