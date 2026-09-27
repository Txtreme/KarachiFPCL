using System.Collections.Generic;
using UnityEngine;

public class MeshCutter : MonoBehaviour
{
    public enum CutMode
    {
        PlaneSweep,
        BoxCollider,
        CustomMeshCollider
    }

    [Header("Cutter Mode")]
    public CutMode cutMode = CutMode.BoxCollider;

    [Header("Continuous Cutting")]
    public bool cutOnTrigger = false; // Set to false so Box and Mesh cuts run continuously in Update

    [Header("Thickness / Margin Tuning")]
    public float bladeThicknessMargin = 0.05f;

    [Header("Plane Sweep Settings (PlaneSweep Mode)")]
    public float cutRadius = 0.5f;

    [Header("Collider References")]
    public BoxCollider targetBoxCollider;
    public MeshCollider targetMeshCollider;

    private Vector3 prevPosition;

    private void Reset()
    {
        targetBoxCollider = GetComponent<BoxCollider>();
        targetMeshCollider = GetComponent<MeshCollider>();
    }

    private void Start()
    {
        prevPosition = transform.position;
        if (targetBoxCollider == null) targetBoxCollider = GetComponent<BoxCollider>();
        if (targetMeshCollider == null) targetMeshCollider = GetComponent<MeshCollider>();
    }

    private void Update()
    {
        Vector3 currentPosition = transform.position;
        float movementSqr = (currentPosition - prevPosition).sqrMagnitude;

        if (cutMode == CutMode.PlaneSweep)
        {
            Vector3 sweepDir = currentPosition - prevPosition;
            float sweepLength = sweepDir.magnitude;

            if (sweepLength > 0.001f)
            {
                Vector3 sweepNorm = sweepDir / sweepLength;
                Vector3 planeNormal = Vector3.Cross(sweepNorm, transform.forward).normalized;
                if (planeNormal == Vector3.zero) planeNormal = transform.up;

                PerformPlaneSweepCut(prevPosition, currentPosition, sweepNorm, sweepLength, planeNormal);
            }
        }
        else if (!cutOnTrigger && movementSqr > 0.00001f) // Execute volume cut only when cutter moved
        {
            PerformVolumeCut();
        }

        prevPosition = currentPosition;
    }

    private void OnTriggerStay(Collider other)
    {
        if (cutOnTrigger && cutMode != CutMode.PlaneSweep)
        {
            FEMPhysicsBody body = other.GetComponentInParent<FEMPhysicsBody>();
            if (body != null)
            {
                CutSingleBodyVolume(body);
            }
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (cutOnTrigger && cutMode != CutMode.PlaneSweep)
        {
            FEMPhysicsBody body = other.GetComponentInParent<FEMPhysicsBody>();
            if (body != null)
            {
                CutSingleBodyVolume(body);
            }
        }
    }

    public void PerformVolumeCut()
    {
        FEMPhysicsBody[] femBodies = Object.FindObjectsByType<FEMPhysicsBody>(
            FindObjectsInactive.Exclude,
            FindObjectsSortMode.None
        );

        foreach (var body in femBodies)
        {
            CutSingleBodyVolume(body);
        }
    }

    public bool CutSingleBodyVolume(FEMPhysicsBody body)
    {
        TetrahedralMesh tetMesh = body.TetMesh;
        Vector3[] positions = body.CurrentPositions;

        if (tetMesh == null || positions == null) return false;

        BoxCollider box = targetBoxCollider != null ? targetBoxCollider : GetComponent<BoxCollider>();
        MeshCollider meshCol = targetMeshCollider != null ? targetMeshCollider : GetComponent<MeshCollider>();

        if (cutMode == CutMode.BoxCollider && box == null) return false;
        if (cutMode == CutMode.CustomMeshCollider && meshCol == null) return false;

        bool meshCutOccurred = false;

        for (int i = 0; i < tetMesh.tets.Count; i++)
        {
            var tet = tetMesh.tets[i];
            if (!tet.active) continue;

            Vector3 p0 = positions[tet.v0];
            Vector3 p1 = positions[tet.v1];
            Vector3 p2 = positions[tet.v2];
            Vector3 p3 = positions[tet.v3];
            Vector3 tetCenter = (p0 + p1 + p2 + p3) * 0.25f;

            bool isInside = false;

            if (cutMode == CutMode.BoxCollider)
            {
                isInside = IsTetInsideBox(p0, p1, p2, p3, tetCenter, box, bladeThicknessMargin);
            }
            else if (cutMode == CutMode.CustomMeshCollider)
            {
                isInside = IsPointInsideMesh(tetCenter, meshCol) ||
                           IsPointInsideMesh(p0, meshCol) ||
                           IsPointInsideMesh(p1, meshCol) ||
                           IsPointInsideMesh(p2, meshCol) ||
                           IsPointInsideMesh(p3, meshCol);
            }

            if (isInside)
            {
                tet.active = false;
                tetMesh.tets[i] = tet;
                meshCutOccurred = true;
            }
        }

        if (meshCutOccurred)
        {
            body.NotifyMeshCut();
        }

        return meshCutOccurred;
    }

    private bool IsTetInsideBox(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 tetCenter, BoxCollider box, float margin)
    {
        Matrix4x4 worldToLocal = box.transform.worldToLocalMatrix;
        Vector3 halfSize = (box.size * 0.5f) + new Vector3(margin, margin, margin);
        Vector3 boxCenter = box.center;

        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4(tetCenter) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4(p0) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4(p1) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4(p2) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4(p3) - boxCenter, halfSize)) return true;

        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p0 + p1) * 0.5f) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p0 + p2) * 0.5f) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p0 + p3) * 0.5f) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p1 + p2) * 0.5f) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p1 + p3) * 0.5f) - boxCenter, halfSize)) return true;
        if (IsLocalPointInBox(worldToLocal.MultiplyPoint3x4((p2 + p3) * 0.5f) - boxCenter, halfSize)) return true;

        return false;
    }

    private bool IsLocalPointInBox(Vector3 localPt, Vector3 halfSize)
    {
        return Mathf.Abs(localPt.x) <= halfSize.x &&
               Mathf.Abs(localPt.y) <= halfSize.y &&
               Mathf.Abs(localPt.z) <= halfSize.z;
    }

    private bool IsPointInsideMesh(Vector3 worldPoint, MeshCollider meshCol)
    {
        if (!meshCol.bounds.Contains(worldPoint)) return false;

        // Directional raycast from outside the bounding box toward the point
        Vector3 start = meshCol.bounds.min - new Vector3(1f, 1f, 1f);
        Vector3 dir = worldPoint - start;
        float dist = dir.magnitude;
        if (dist < 0.0001f) return false;
        dir /= dist;

        RaycastHit[] hits = Physics.RaycastAll(start, dir, dist);
        int hitCount = 0;
        for (int i = 0; i < hits.Length; i++)
        {
            if (hits[i].collider == meshCol)
            {
                hitCount++;
            }
        }

        // An odd number of surface intersections indicates the point is inside the closed mesh
        return (hitCount % 2) == 1;
    }

    public void PerformPlaneSweepCut(Vector3 startPt, Vector3 endPt, Vector3 sweepNorm, float sweepLength, Vector3 planeNormal)
    {
        FEMPhysicsBody[] femBodies = Object.FindObjectsByType<FEMPhysicsBody>(
            FindObjectsInactive.Exclude,
            FindObjectsSortMode.None
        );

        foreach (var body in femBodies)
        {
            TetrahedralMesh tetMesh = body.TetMesh;
            Vector3[] positions = body.CurrentPositions;

            if (tetMesh == null || positions == null) continue;

            bool meshCutOccurred = false;

            for (int i = 0; i < tetMesh.tets.Count; i++)
            {
                var tet = tetMesh.tets[i];
                if (!tet.active) continue;

                Vector3 p0 = positions[tet.v0];
                Vector3 p1 = positions[tet.v1];
                Vector3 p2 = positions[tet.v2];
                Vector3 p3 = positions[tet.v3];
                Vector3 tetCenter = (p0 + p1 + p2 + p3) * 0.25f;

                float proj = Vector3.Dot(tetCenter - startPt, sweepNorm);
                proj = Mathf.Clamp(proj, 0f, sweepLength);
                Vector3 closestPointOnSweep = startPt + sweepNorm * proj;

                if (Vector3.Distance(tetCenter, closestPointOnSweep) > cutRadius) continue;

                float d0 = Vector3.Dot(p0 - closestPointOnSweep, planeNormal);
                float d1 = Vector3.Dot(p1 - closestPointOnSweep, planeNormal);
                float d2 = Vector3.Dot(p2 - closestPointOnSweep, planeNormal);
                float d3 = Vector3.Dot(p3 - closestPointOnSweep, planeNormal);

                bool holdsPos = d0 > 0 || d1 > 0 || d2 > 0 || d3 > 0;
                bool holdsNeg = d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0;

                if (holdsPos && holdsNeg)
                {
                    tet.active = false;
                    tetMesh.tets[i] = tet;
                    meshCutOccurred = true;
                }
            }

            if (meshCutOccurred)
            {
                body.NotifyMeshCut();
            }
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (cutMode == CutMode.PlaneSweep)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(transform.position, cutRadius);
        }
    }
}