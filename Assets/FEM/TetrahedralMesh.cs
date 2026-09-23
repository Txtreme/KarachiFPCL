using System;
using System.Collections.Generic;
using UnityEngine;

public class TetrahedralMesh
{
    public List<Vector3> vertices = new List<Vector3>();
    public List<Tetrahedron> tets = new List<Tetrahedron>();

    public struct Tetrahedron
    {
        public int v0, v1, v2, v3;
        public float restVolume;
        public bool active;

        public Tetrahedron(int v0, int v1, int v2, int v3, float restVolume)
        {
            this.v0 = v0;
            this.v1 = v1;
            this.v2 = v2;
            this.v3 = v3;
            this.restVolume = restVolume;
            this.active = true;
        }
    }

    public struct FaceKey : IEquatable<FaceKey>
    {
        public readonly int a, b, c;

        public FaceKey(int v0, int v1, int v2)
        {
            int[] idx = { v0, v1, v2 };
            Array.Sort(idx);
            a = idx[0];
            b = idx[1];
            c = idx[2];
        }

        public bool Equals(FaceKey other) => a == other.a && b == other.b && c == other.c;
        public override bool Equals(object obj) => obj is FaceKey other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(a, b, c);
    }

    public static TetrahedralMesh GenerateFromMesh(Mesh sourceMesh, Vector3Int gridRes)
    {
        TetrahedralMesh tetMesh = new TetrahedralMesh();
        Bounds bounds = sourceMesh.bounds;
        Vector3 min = bounds.min;
        Vector3 size = bounds.size;
        Vector3 cellSize = new Vector3(size.x / gridRes.x, size.y / gridRes.y, size.z / gridRes.z);

        Vector3[] srcVerts = sourceMesh.vertices;
        int[] srcTris = sourceMesh.triangles;

        Dictionary<Vector3Int, int> nodeMap = new Dictionary<Vector3Int, int>();

        int GetOrAddVertex(Vector3Int gridPos)
        {
            if (nodeMap.TryGetValue(gridPos, out int index)) return index;

            Vector3 worldPos = min + new Vector3(gridPos.x * cellSize.x, gridPos.y * cellSize.y, gridPos.z * cellSize.z);
            tetMesh.vertices.Add(worldPos);
            index = tetMesh.vertices.Count - 1;
            nodeMap[gridPos] = index;
            return index;
        }

        for (int x = 0; x < gridRes.x; x++)
        {
            for (int y = 0; y < gridRes.y; y++)
            {
                for (int z = 0; z < gridRes.z; z++)
                {
                    Vector3Int p000 = new Vector3Int(x, y, z);
                    Vector3Int p100 = new Vector3Int(x + 1, y, z);
                    Vector3Int p010 = new Vector3Int(x, y + 1, z);
                    Vector3Int p110 = new Vector3Int(x + 1, y + 1, z);
                    Vector3Int p001 = new Vector3Int(x, y, z + 1);
                    Vector3Int p101 = new Vector3Int(x + 1, y, z + 1);
                    Vector3Int p011 = new Vector3Int(x, y + 1, z + 1);
                    Vector3Int p111 = new Vector3Int(x + 1, y + 1, z + 1);

                    Vector3Int[][] cellTets = new Vector3Int[][]
                    {
                        new Vector3Int[] { p000, p100, p010, p001 },
                        new Vector3Int[] { p100, p110, p010, p111 },
                        new Vector3Int[] { p100, p001, p010, p101 },
                        new Vector3Int[] { p010, p001, p111, p011 },
                        new Vector3Int[] { p100, p010, p111, p001 }
                    };

                    foreach (var rawTet in cellTets)
                    {
                        int v0 = GetOrAddVertex(rawTet[0]);
                        int v1 = GetOrAddVertex(rawTet[1]);
                        int v2 = GetOrAddVertex(rawTet[2]);
                        int v3 = GetOrAddVertex(rawTet[3]);

                        Vector3 center = (tetMesh.vertices[v0] + tetMesh.vertices[v1] + tetMesh.vertices[v2] + tetMesh.vertices[v3]) * 0.25f;

                        if (IsPointInsideMesh(center, srcVerts, srcTris))
                        {
                            float vol = CalculateSignedVolume(tetMesh.vertices[v0], tetMesh.vertices[v1], tetMesh.vertices[v2], tetMesh.vertices[v3]);
                            if (vol < 0) { int tmp = v1; v1 = v2; v2 = tmp; vol = -vol; }
                            if (vol > 1e-5f)
                            {
                                tetMesh.tets.Add(new Tetrahedron(v0, v1, v2, v3, vol));
                            }
                        }
                    }
                }
            }
        }

        tetMesh.ProjectBoundaryVerticesToSurface(srcVerts, srcTris, cellSize.magnitude * 0.5f);
        return tetMesh;
    }

    private void ProjectBoundaryVerticesToSurface(Vector3[] srcVerts, int[] srcTris, float maxSnapDist)
    {
        List<int> surfaceTriangles = ReconstructSurfaceTriangles();
        HashSet<int> boundaryVertIndices = new HashSet<int>(surfaceTriangles);

        foreach (int vIdx in boundaryVertIndices)
        {
            Vector3 origPos = vertices[vIdx];
            Vector3 closestPoint = origPos;
            float minSqrDist = float.MaxValue;

            for (int i = 0; i < srcTris.Length; i += 3)
            {
                Vector3 a = srcVerts[srcTris[i]];
                Vector3 b = srcVerts[srcTris[i + 1]];
                Vector3 c = srcVerts[srcTris[i + 2]];

                Vector3 pt = ClosestPointOnTriangle(origPos, a, b, c);
                float sqrDist = (pt - origPos).sqrMagnitude;

                if (sqrDist < minSqrDist)
                {
                    minSqrDist = sqrDist;
                    closestPoint = pt;
                }
            }

            if (Mathf.Sqrt(minSqrDist) <= maxSnapDist)
            {
                Vector3 targetPos = Vector3.Lerp(origPos, closestPoint, 0.7f);
                vertices[vIdx] = targetPos;

                bool validSnap = true;
                foreach (var tet in tets)
                {
                    if (tet.v0 == vIdx || tet.v1 == vIdx || tet.v2 == vIdx || tet.v3 == vIdx)
                    {
                        float vol = CalculateSignedVolume(vertices[tet.v0], vertices[tet.v1], vertices[tet.v2], vertices[tet.v3]);
                        if (vol <= 1e-5f)
                        {
                            validSnap = false;
                            break;
                        }
                    }
                }

                if (!validSnap)
                {
                    vertices[vIdx] = origPos;
                }
            }
        }

        for (int i = 0; i < tets.Count; i++)
        {
            var tet = tets[i];
            float vol = CalculateSignedVolume(vertices[tet.v0], vertices[tet.v1], vertices[tet.v2], vertices[tet.v3]);
            tet.restVolume = Mathf.Max(vol, 1e-5f);
            tets[i] = tet;
        }
    }

    private static Vector3 ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;

        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return a;

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) return b;

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        {
            float v = d1 / (d1 - d3);
            return a + v * ab;
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) return c;

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        {
            float w = d2 / (d2 - d6);
            return a + w * ac;
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b);
        }

        float denom = 1.0f / (va + vb + vc);
        float v_in = vb * denom;
        float w_in = vc * denom;
        return a + ab * v_in + ac * w_in;
    }

    public static float CalculateSignedVolume(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        return Vector3.Dot(p1 - p0, Vector3.Cross(p2 - p0, p3 - p0)) / 6.0f;
    }

    private static bool IsPointInsideMesh(Vector3 point, Vector3[] verts, int[] tris)
    {
        int hitsX = CountRayIntersections(point, Vector3.right, verts, tris);
        int hitsY = CountRayIntersections(point, Vector3.up, verts, tris);
        int hitsZ = CountRayIntersections(point, Vector3.forward, verts, tris);

        int insideVotes = 0;
        if ((hitsX % 2) == 1) insideVotes++;
        if ((hitsY % 2) == 1) insideVotes++;
        if ((hitsZ % 2) == 1) insideVotes++;

        return insideVotes >= 2;
    }

    private static int CountRayIntersections(Vector3 origin, Vector3 dir, Vector3[] verts, int[] tris)
    {
        int count = 0;
        for (int i = 0; i < tris.Length; i += 3)
        {
            if (RayTriangleIntersection(origin, dir, verts[tris[i]], verts[tris[i + 1]], verts[tris[i + 2]]))
            {
                count++;
            }
        }
        return count;
    }

    private static bool RayTriangleIntersection(Vector3 origin, Vector3 dir, Vector3 v0, Vector3 v1, Vector3 v2)
    {
        const float EPSILON = 1e-6f;
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 h = Vector3.Cross(dir, edge2);
        float a = Vector3.Dot(edge1, h);

        if (a > -EPSILON && a < EPSILON) return false;

        float f = 1.0f / a;
        Vector3 s = origin - v0;
        float u = f * Vector3.Dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;

        Vector3 q = Vector3.Cross(s, edge1);
        float v = f * Vector3.Dot(dir, q);
        if (v < 0.0f || u + v > 1.0f) return false;

        float t = f * Vector3.Dot(edge2, q);
        return t > EPSILON;
    }

    public List<int> ReconstructSurfaceTriangles()
    {
        Dictionary<(int, int, int), int> faceCounts = new Dictionary<(int, int, int), int>();
        Dictionary<(int, int, int), (int, int, int)> faceOrientations = new Dictionary<(int, int, int), (int, int, int)>();

        if (tets == null) return new List<int>();

        foreach (var tet in tets)
        {
            if (!tet.active) continue;

            (int, int, int)[] faces = new (int, int, int)[]
            {
            (tet.v0, tet.v2, tet.v1),
            (tet.v0, tet.v1, tet.v3),
            (tet.v0, tet.v3, tet.v2),
            (tet.v1, tet.v2, tet.v3)
            };

            foreach (var f in faces)
            {
                int[] sorted = new int[] { f.Item1, f.Item2, f.Item3 };
                System.Array.Sort(sorted);
                var key = (sorted[0], sorted[1], sorted[2]);

                if (faceCounts.ContainsKey(key))
                {
                    faceCounts[key]++;
                }
                else
                {
                    faceCounts[key] = 1;
                    faceOrientations[key] = f;
                }
            }
        }

        List<int> surfaceTris = new List<int>();
        foreach (var kvp in faceCounts)
        {
            if (kvp.Value == 1)
            {
                var f = faceOrientations[kvp.Key];
                surfaceTris.Add(f.Item1);
                surfaceTris.Add(f.Item2);
                surfaceTris.Add(f.Item3);
            }
        }

        return surfaceTris;
    }
}