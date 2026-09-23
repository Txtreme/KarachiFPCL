using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class FEMVisualShell : MonoBehaviour
{
    public enum ShellMode
    {
        ProceduralSurface,
        CustomMeshBarycentric
    }

    [Header("Shell Configuration")]
    public FEMPhysicsBody physicsBody;
    public ShellMode shellMode = ShellMode.ProceduralSurface;

    [Header("PBR Texture Mapping")]
    public Vector2 uvScale = new Vector2(1f, 1f);
    public Vector2 uvOffset = new Vector2(0f, 0f);

    [Header("Custom High-Poly Model (Custom Mesh Mode)")]
    public Mesh customVisualMesh;

    private Mesh shellMesh;
    private MeshFilter meshFilter;
    private bool isInitialized = false;

    // Barycentric Skinning Data
    private Vector3[] customSourceVertices;
    private Vector3[] customSourceNormals;
    private Vector4[] customSourceTangents;
    private Vector2[] customSourceUVs;
    private int[] customTetIndices;
    private Vector4[] customBaryWeights;

    private void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        if (physicsBody == null)
            physicsBody = GetComponentInParent<FEMPhysicsBody>();
    }

    private void Start()
    {
        RebuildShell();
    }

    public void OnMeshCut()
    {
        isInitialized = false;
        RebuildShell();
    }

    public void RebuildShell()
    {
        if (physicsBody == null || physicsBody.TetMesh == null || physicsBody.CurrentPositions == null) return;

        if (shellMesh == null)
        {
            shellMesh = new Mesh { name = "FEM_Visual_Shell" };
            shellMesh.MarkDynamic();
            if (meshFilter != null) meshFilter.mesh = shellMesh;
        }

        if (shellMode == ShellMode.CustomMeshBarycentric && customVisualMesh != null)
        {
            SetupBarycentricSkinning();
        }
        else
        {
            SetupProceduralSurfaceShell();
        }

        isInitialized = true;
    }

    private void SetupProceduralSurfaceShell()
    {
        List<int> tris = physicsBody.TetMesh.ReconstructSurfaceTriangles();
        Vector3[] physPositions = physicsBody.CurrentPositions;

        if (physPositions == null || physPositions.Length == 0) return;

        // Fallback: If no boundary triangles detected, render all tet faces so something is always visible
        if (tris == null || tris.Count == 0)
        {
            Debug.LogWarning("FEMVisualShell: ReconstructSurfaceTriangles returned 0 surface faces. Using all tet faces fallback.");
            tris = GetAllTetFaces(physicsBody.TetMesh);
        }

        Vector3[] localVerts = new Vector3[physPositions.Length];
        for (int i = 0; i < physPositions.Length; i++)
        {
            localVerts[i] = transform.InverseTransformPoint(physPositions[i]);
        }

        shellMesh.Clear();
        shellMesh.vertices = localVerts;
        shellMesh.triangles = tris.ToArray();
        shellMesh.RecalculateNormals();

        Vector3[] normals = shellMesh.normals;
        Vector2[] uvs = new Vector2[localVerts.Length];

        for (int i = 0; i < localVerts.Length; i++)
        {
            Vector3 pos = localVerts[i];
            Vector3 n = (normals != null && i < normals.Length) ? normals[i] : Vector3.up;

            float absX = Mathf.Abs(n.x);
            float absY = Mathf.Abs(n.y);
            float absZ = Mathf.Abs(n.z);

            Vector2 projectedUV;
            if (absY >= absX && absY >= absZ)
                projectedUV = new Vector2(pos.x, pos.z);
            else if (absX >= absY && absX >= absZ)
                projectedUV = new Vector2(pos.z, pos.y);
            else
                projectedUV = new Vector2(pos.x, pos.y);

            uvs[i] = new Vector2(
                projectedUV.x * uvScale.x + uvOffset.x,
                projectedUV.y * uvScale.y + uvOffset.y
            );
        }

        shellMesh.uv = uvs;
        shellMesh.RecalculateTangents();
        shellMesh.RecalculateBounds();
    }

    private List<int> GetAllTetFaces(TetrahedralMesh tetMesh)
    {
        List<int> faces = new List<int>();
        if (tetMesh == null || tetMesh.tets == null) return faces;

        foreach (var tet in tetMesh.tets)
        {
            if (!tet.active) continue;
            faces.Add(tet.v0); faces.Add(tet.v2); faces.Add(tet.v1);
            faces.Add(tet.v0); faces.Add(tet.v1); faces.Add(tet.v3);
            faces.Add(tet.v0); faces.Add(tet.v3); faces.Add(tet.v2);
            faces.Add(tet.v1); faces.Add(tet.v2); faces.Add(tet.v3);
        }
        return faces;
    }

    private void SetupBarycentricSkinning()
    {
        var tetMesh = physicsBody.TetMesh;
        if (tetMesh == null || tetMesh.tets.Count == 0) return;

        customSourceVertices = customVisualMesh.vertices;
        customSourceNormals = customVisualMesh.normals;
        customSourceTangents = customVisualMesh.tangents;
        customSourceUVs = customVisualMesh.uv;

        int numVerts = customSourceVertices.Length;
        customTetIndices = new int[numVerts];
        customBaryWeights = new Vector4[numVerts];

        for (int i = 0; i < numVerts; i++)
        {
            int bestTetIdx = 0;
            Vector4 bestWeights = Vector4.zero;
            float minDistance = float.MaxValue;

            for (int t = 0; t < tetMesh.tets.Count; t++)
            {
                var tet = tetMesh.tets[t];
                if (!tet.active) continue;

                Vector3 x0 = tetMesh.vertices[tet.v0];
                Vector3 x1 = tetMesh.vertices[tet.v1];
                Vector3 x2 = tetMesh.vertices[tet.v2];
                Vector3 x3 = tetMesh.vertices[tet.v3];

                Matrix3x3 Dm = new Matrix3x3(x1 - x0, x2 - x0, x3 - x0);
                Matrix3x3 invDm = Dm.Inverse();

                Vector3 diff = customSourceVertices[i] - x0;
                Vector3 alphaBetaGamma = invDm.MultiplyVector(diff);

                float w1 = alphaBetaGamma.x;
                float w2 = alphaBetaGamma.y;
                float w3 = alphaBetaGamma.z;
                float w0 = 1.0f - (w1 + w2 + w3);

                Vector3 tetCenter = (x0 + x1 + x2 + x3) * 0.25f;
                float dist = (customSourceVertices[i] - tetCenter).sqrMagnitude;

                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestTetIdx = t;
                    bestWeights = new Vector4(w0, w1, w2, w3);
                }
            }

            customTetIndices[i] = bestTetIdx;
            customBaryWeights[i] = bestWeights;
        }

        shellMesh.Clear();
        shellMesh.vertices = new Vector3[numVerts];
        shellMesh.triangles = customVisualMesh.triangles;
        shellMesh.uv = customSourceUVs;
        if (customSourceNormals != null && customSourceNormals.Length > 0) shellMesh.normals = customSourceNormals;
        if (customSourceTangents != null && customSourceTangents.Length > 0) shellMesh.tangents = customSourceTangents;
    }

    private void LateUpdate()
    {
        if (!isInitialized || shellMesh == null || shellMesh.triangles.Length == 0)
        {
            RebuildShell();
            return;
        }

        UpdateShellPositions();
    }

    private void UpdateShellPositions()
    {
        if (physicsBody == null || physicsBody.CurrentPositions == null) return;

        Vector3[] physPositions = physicsBody.CurrentPositions;

        if (shellMode == ShellMode.ProceduralSurface)
        {
            Vector3[] localVerts = new Vector3[physPositions.Length];
            for (int i = 0; i < physPositions.Length; i++)
            {
                localVerts[i] = transform.InverseTransformPoint(physPositions[i]);
            }

            shellMesh.vertices = localVerts;
            shellMesh.RecalculateNormals();
            shellMesh.RecalculateBounds();
        }
        else if (shellMode == ShellMode.CustomMeshBarycentric && customTetIndices != null)
        {
            var tetMesh = physicsBody.TetMesh;
            Vector3[] deformedVerts = new Vector3[customSourceVertices.Length];

            for (int i = 0; i < customSourceVertices.Length; i++)
            {
                var tet = tetMesh.tets[customTetIndices[i]];
                Vector4 w = customBaryWeights[i];

                Vector3 p0 = physPositions[tet.v0];
                Vector3 p1 = physPositions[tet.v1];
                Vector3 p2 = physPositions[tet.v2];
                Vector3 p3 = physPositions[tet.v3];

                Vector3 worldPos = w.x * p0 + w.y * p1 + w.z * p2 + w.w * p3;
                deformedVerts[i] = transform.InverseTransformPoint(worldPos);
            }

            shellMesh.vertices = deformedVerts;
            shellMesh.RecalculateNormals();
            shellMesh.RecalculateBounds();
        }
    }
}