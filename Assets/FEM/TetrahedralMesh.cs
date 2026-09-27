using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public struct Tetrahedron
{
    public int v0, v1, v2, v3;
    public float restVolume;
    public bool active;
    public int materialId;

    public Tetrahedron(int v0, int v1, int v2, int v3)
    {
        this.v0 = v0;
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
        this.restVolume = 0f;
        this.active = true;
        this.materialId = 0;
    }
}

public class TetrahedralMesh
{
    public List<Vector3> vertices = new List<Vector3>();
    public List<Tetrahedron> tets = new List<Tetrahedron>();

    public static TetrahedralMesh GenerateFromMesh(Mesh mesh, Vector3Int gridRes)
    {
        TetrahedralMesh tetMesh = new TetrahedralMesh();
        if (mesh == null) return tetMesh;

        Vector3[] origVerts = mesh.vertices;
        int[] origTris = mesh.triangles;
        if (origVerts.Length == 0 || origTris.Length == 0) return tetMesh;

        // Spatial acceleration wrapper for fast point testing & projection
        MeshSpatialAccelerator accelerator = new MeshSpatialAccelerator(origVerts, origTris);

        Bounds bounds = mesh.bounds;
        bounds.Expand(0.01f);

        int nx = Mathf.Max(2, gridRes.x);
        int ny = Mathf.Max(2, gridRes.y);
        int nz = Mathf.Max(2, gridRes.z);

        Vector3 step = new Vector3(
            bounds.size.x / (nx - 1),
            bounds.size.y / (ny - 1),
            bounds.size.z / (nz - 1)
        );

        Vector3[,,] gridVerts = new Vector3[nx, ny, nz];
        for (int x = 0; x < nx; x++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int z = 0; z < nz; z++)
                {
                    gridVerts[x, y, z] = bounds.min + new Vector3(x * step.x, y * step.y, z * step.z);
                }
            }
        }

        Dictionary<Vector3Int, int> gridToCompactIndex = new Dictionary<Vector3Int, int>();

        int GetOrAddVertex(Vector3Int cell)
        {
            if (gridToCompactIndex.TryGetValue(cell, out int idx)) return idx;
            idx = tetMesh.vertices.Count;
            tetMesh.vertices.Add(gridVerts[cell.x, cell.y, cell.z]);
            gridToCompactIndex[cell] = idx;
            return idx;
        }

        for (int x = 0; x < nx - 1; x++)
        {
            for (int y = 0; y < ny - 1; y++)
            {
                for (int z = 0; z < nz - 1; z++)
                {
                    Vector3Int c0 = new Vector3Int(x, y, z);
                    Vector3Int c1 = new Vector3Int(x + 1, y, z);
                    Vector3Int c2 = new Vector3Int(x + 1, y + 1, z);
                    Vector3Int c3 = new Vector3Int(x, y + 1, z);
                    Vector3Int c4 = new Vector3Int(x, y, z + 1);
                    Vector3Int c5 = new Vector3Int(x + 1, y, z + 1);
                    Vector3Int c6 = new Vector3Int(x + 1, y + 1, z + 1);
                    Vector3Int c7 = new Vector3Int(x, y + 1, z + 1);

                    Vector3Int[][] cellTetsPattern;

                    if ((x + y + z) % 2 == 0)
                    {
                        cellTetsPattern = new Vector3Int[][]
                        {
                            new Vector3Int[] { c0, c1, c3, c4 },
                            new Vector3Int[] { c1, c2, c3, c6 },
                            new Vector3Int[] { c1, c4, c5, c6 },
                            new Vector3Int[] { c3, c4, c6, c7 },
                            new Vector3Int[] { c1, c3, c4, c6 }
                        };
                    }
                    else
                    {
                        cellTetsPattern = new Vector3Int[][]
                        {
                            new Vector3Int[] { c0, c1, c2, c5 },
                            new Vector3Int[] { c0, c2, c3, c7 },
                            new Vector3Int[] { c0, c4, c5, c7 },
                            new Vector3Int[] { c2, c5, c6, c7 },
                            new Vector3Int[] { c0, c2, c5, c7 }
                        };
                    }

                    foreach (var tetPattern in cellTetsPattern)
                    {
                        Vector3 p0 = gridVerts[tetPattern[0].x, tetPattern[0].y, tetPattern[0].z];
                        Vector3 p1 = gridVerts[tetPattern[1].x, tetPattern[1].y, tetPattern[1].z];
                        Vector3 p2 = gridVerts[tetPattern[2].x, tetPattern[2].y, tetPattern[2].z];
                        Vector3 p3 = gridVerts[tetPattern[3].x, tetPattern[3].y, tetPattern[3].z];

                        Vector3 centroid = (p0 + p1 + p2 + p3) * 0.25f;

                        if (accelerator.IsPointInsideMesh(centroid) ||
                            accelerator.IsPointInsideMesh(p0) ||
                            accelerator.IsPointInsideMesh(p1) ||
                            accelerator.IsPointInsideMesh(p2) ||
                            accelerator.IsPointInsideMesh(p3))
                        {
                            int i0 = GetOrAddVertex(tetPattern[0]);
                            int i1 = GetOrAddVertex(tetPattern[1]);
                            int i2 = GetOrAddVertex(tetPattern[2]);
                            int i3 = GetOrAddVertex(tetPattern[3]);

                            tetMesh.tets.Add(new Tetrahedron(i0, i1, i2, i3));
                        }
                    }
                }
            }
        }

        // =========================================================================
        // SMOOTHING STEP: Project all surface boundary vertices to closest mesh point
        // =========================================================================
        List<int> surfaceTris = tetMesh.ReconstructSurfaceTriangles();
        HashSet<int> surfaceVertIndices = new HashSet<int>(surfaceTris);

        foreach (int vIdx in surfaceVertIndices)
        {
            tetMesh.vertices[vIdx] = accelerator.GetClosestPointOnMesh(tetMesh.vertices[vIdx]);
        }

        return tetMesh;
    }

    // =========================================================================
    // SPATIAL ACCELERATOR CLASS
    // =========================================================================
    private class MeshSpatialAccelerator
    {
        private struct TriangleData
        {
            public Vector3 v0, v1, v2;
            public Bounds bounds;
        }

        private readonly TriangleData[] triangles;
        private readonly Bounds totalBounds;

        public MeshSpatialAccelerator(Vector3[] verts, int[] tris)
        {
            int triCount = tris.Length / 3;
            triangles = new TriangleData[triCount];

            if (verts.Length > 0)
            {
                totalBounds = new Bounds(verts[0], Vector3.zero);
            }

            for (int i = 0; i < triCount; i++)
            {
                Vector3 v0 = verts[tris[i * 3]];
                Vector3 v1 = verts[tris[i * 3 + 1]];
                Vector3 v2 = verts[tris[i * 3 + 2]];

                triangles[i].v0 = v0;
                triangles[i].v1 = v1;
                triangles[i].v2 = v2;

                Bounds b = new Bounds(v0, Vector3.zero);
                b.Encapsulate(v1);
                b.Encapsulate(v2);
                triangles[i].bounds = b;

                totalBounds.Encapsulate(b);
            }

            totalBounds.Expand(0.02f);
        }

        public bool IsPointInsideMesh(Vector3 point)
        {
            if (!totalBounds.Contains(point)) return false;

            Vector3 rayDir = new Vector3(0.408248f, 0.816497f, 0.408248f);
            int intersections = 0;

            for (int i = 0; i < triangles.Length; i++)
            {
                if (RayIntersectsAABB(point, rayDir, triangles[i].bounds))
                {
                    if (RayTriangleIntersection(point, rayDir, triangles[i].v0, triangles[i].v1, triangles[i].v2))
                    {
                        intersections++;
                    }
                }
            }

            return (intersections % 2) == 1;
        }

        public Vector3 GetClosestPointOnMesh(Vector3 point)
        {
            float minSqDist = float.MaxValue;
            Vector3 closestPoint = point;

            for (int i = 0; i < triangles.Length; i++)
            {
                float sqDistToBounds = triangles[i].bounds.SqrDistance(point);
                if (sqDistToBounds >= minSqDist) continue;

                Vector3 p = ClosestPointOnTriangle(point, triangles[i].v0, triangles[i].v1, triangles[i].v2);
                float sqDist = (point - p).sqrMagnitude;
                if (sqDist < minSqDist)
                {
                    minSqDist = sqDist;
                    closestPoint = p;
                }
            }
            return closestPoint;
        }

        private static bool RayIntersectsAABB(Vector3 origin, Vector3 dir, Bounds bounds)
        {
            Vector3 min = bounds.min;
            Vector3 max = bounds.max;

            float tmin = (min.x - origin.x) / dir.x;
            float tmax = (max.x - origin.x) / dir.x;
            if (tmin > tmax) { float tmp = tmin; tmin = tmax; tmax = tmp; }

            float tymin = (min.y - origin.y) / dir.y;
            float tymax = (max.y - origin.y) / dir.y;
            if (tymin > tymax) { float tmp = tymin; tymin = tymax; tymax = tmp; }

            if ((tmin > tymax) || (tymin > tmax)) return false;
            if (tymin > tmin) tmin = tymin;
            if (tymax < tmax) tmax = tymax;

            float tzmin = (min.z - origin.z) / dir.z;
            float tzmax = (max.z - origin.z) / dir.z;
            if (tzmin > tzmax) { float tmp = tzmin; tzmin = tzmax; tzmax = tmp; }

            if ((tmin > tzmax) || (tzmin > tmax)) return false;
            if (tzmax < tmax) tmax = tzmax;

            return tmax >= 0f;
        }

        private static bool RayTriangleIntersection(Vector3 rayOrigin, Vector3 rayDir, Vector3 v0, Vector3 v1, Vector3 v2)
        {
            const float EPSILON = 1e-7f;
            Vector3 edge1 = v1 - v0;
            Vector3 edge2 = v2 - v0;
            Vector3 h = Vector3.Cross(rayDir, edge2);
            float a = Vector3.Dot(edge1, h);

            if (a > -EPSILON && a < EPSILON) return false;

            float f = 1.0f / a;
            Vector3 s = rayOrigin - v0;
            float u = f * Vector3.Dot(s, h);

            if (u < 0.0f || u > 1.0f) return false;

            Vector3 q = Vector3.Cross(s, edge1);
            float v = f * Vector3.Dot(rayDir, q);

            if (v < 0.0f || u + v > 1.0f) return false;

            float t = f * Vector3.Dot(edge2, q);
            return t > EPSILON;
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
            float v_ = vb * denom;
            float w_ = vc * denom;
            return a + ab * v_ + ac * w_;
        }
    }

    public List<int> ReconstructSurfaceTriangles()
    {
        List<int> surfaceTriangles = new List<int>();
        if (tets == null || tets.Count == 0) return surfaceTriangles;

        Dictionary<TriangleKey, TriangleFace> faceCounts = new Dictionary<TriangleKey, TriangleFace>();

        void AddFace(int v0, int v1, int v2)
        {
            TriangleKey key = new TriangleKey(v0, v1, v2);
            if (faceCounts.TryGetValue(key, out TriangleFace face))
            {
                face.count++;
            }
            else
            {
                faceCounts[key] = new TriangleFace(v0, v1, v2);
            }
        }

        for (int i = 0; i < tets.Count; i++)
        {
            var tet = tets[i];
            if (!tet.active) continue;

            AddFace(tet.v0, tet.v2, tet.v1);
            AddFace(tet.v0, tet.v1, tet.v3);
            AddFace(tet.v0, tet.v3, tet.v2);
            AddFace(tet.v1, tet.v2, tet.v3);
        }

        foreach (var pair in faceCounts.Values)
        {
            if (pair.count == 1)
            {
                surfaceTriangles.Add(pair.v0);
                surfaceTriangles.Add(pair.v1);
                surfaceTriangles.Add(pair.v2);
            }
        }

        return surfaceTriangles;
    }

    private struct TriangleKey
    {
        public readonly int a, b, c;

        public TriangleKey(int v0, int v1, int v2)
        {
            if (v0 < v1)
            {
                if (v0 < v2)
                {
                    a = v0;
                    if (v1 < v2) { b = v1; c = v2; }
                    else { b = v2; c = v1; }
                }
                else { a = v2; b = v0; c = v1; }
            }
            else
            {
                if (v1 < v2)
                {
                    a = v1;
                    if (v0 < v2) { b = v0; c = v2; }
                    else { b = v2; c = v0; }
                }
                else { a = v2; b = v1; c = v0; }
            }
        }

        public override bool Equals(object obj)
        {
            if (!(obj is TriangleKey)) return false;
            TriangleKey other = (TriangleKey)obj;
            return a == other.a && b == other.b && c == other.c;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + a;
                hash = hash * 31 + b;
                hash = hash * 31 + c;
                return hash;
            }
        }
    }

    private class TriangleFace
    {
        public int v0, v1, v2;
        public int count;

        public TriangleFace(int v0, int v1, int v2)
        {
            this.v0 = v0;
            this.v1 = v1;
            this.v2 = v2;
            this.count = 1;
        }
    }
}