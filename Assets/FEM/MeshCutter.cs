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

    [Header("Continuous Cutting Settings")]
    public bool cutOnTrigger = true;
    [Tooltip("Cooldown in seconds between cuts on the same body to prevent frame-by-frame cascade destruction.")]
    public float cutCooldown = 0.35f;

    [Header("Thickness / Margin Tuning")]
    [Tooltip("Margin around cutter box for vertex inclusion.")]
    public float bladeThicknessMargin = 0.005f;

    [Header("Plane Sweep Settings (PlaneSweep Mode)")]
    public float cutRadius = 0.15f;
    [Tooltip("Which local axis of the cutter represents the flat blade normal? (Default: Up)")]
    public Vector3 bladePlaneNormalAxis = Vector3.up;

    [Header("Collider References")]
    public BoxCollider targetBoxCollider;
    public MeshCollider targetMeshCollider;

    private Vector3 prevPosition;

    private static readonly List<FEMPhysicsBody> activeBodies = new List<FEMPhysicsBody>();
    private static readonly RaycastHit[] raycastHitBuffer = new RaycastHit[128];
    private static readonly Dictionary<FEMPhysicsBody, float> lastCutTimes = new Dictionary<FEMPhysicsBody, float>();

    public static void RegisterBody(FEMPhysicsBody body)
    {
        if (body != null && !activeBodies.Contains(body))
        {
            activeBodies.Add(body);
        }
    }

    public static void UnregisterBody(FEMPhysicsBody body)
    {
        if (body != null)
        {
            activeBodies.Remove(body);
            lastCutTimes.Remove(body);
        }
    }

    public static void RecordCutForBody(FEMPhysicsBody body)
    {
        if (body != null)
        {
            lastCutTimes[body] = Time.time;
        }
    }

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

        RefreshBodyCache();
    }

    private void RefreshBodyCache()
    {
        if (activeBodies.Count == 0)
        {
            FEMPhysicsBody[] femBodies = Object.FindObjectsByType<FEMPhysicsBody>(
                FindObjectsInactive.Exclude,
                FindObjectsSortMode.None
            );
            foreach (var body in femBodies)
            {
                RegisterBody(body);
            }
        }
    }

    public bool CanCutBody(FEMPhysicsBody body)
    {
        if (body == null) return false;
        if (lastCutTimes.TryGetValue(body, out float lastTime))
        {
            if (Time.time - lastTime < cutCooldown) return false;
        }
        return true;
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
                Vector3 planeNormal = transform.TransformDirection(bladePlaneNormalAxis).normalized;

                PerformPlaneSweepCut(prevPosition, currentPosition, sweepNorm, sweepLength, planeNormal);
            }
        }
        else if (!cutOnTrigger && movementSqr > 0.00001f)
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
        activeBodies.RemoveAll(b => b == null);
        if (activeBodies.Count == 0) RefreshBodyCache();

        for (int i = activeBodies.Count - 1; i >= 0; i--)
        {
            CutSingleBodyVolume(activeBodies[i]);
        }
    }

    public bool CutSingleBodyVolume(FEMPhysicsBody body)
    {
        if (body == null || !CanCutBody(body)) return false;

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

            if (tet.v0 >= positions.Length || tet.v1 >= positions.Length ||
                tet.v2 >= positions.Length || tet.v3 >= positions.Length) continue;

            Vector3 p0 = positions[tet.v0];
            Vector3 p1 = positions[tet.v1];
            Vector3 p2 = positions[tet.v2];
            Vector3 p3 = positions[tet.v3];
            Vector3 tetCenter = (p0 + p1 + p2 + p3) * 0.25f;

            bool isInside = false;

            if (cutMode == CutMode.BoxCollider)
            {
                isInside = IsTetIntersectingBox(p0, p1, p2, p3, tetCenter, box, bladeThicknessMargin);
            }
            else if (cutMode == CutMode.CustomMeshCollider)
            {
                if (meshCol.bounds.Contains(tetCenter))
                {
                    isInside = IsPointInsideMesh(tetCenter, meshCol);
                }
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
            RecordCutForBody(body);
            body.NotifyMeshCut();
        }

        return meshCutOccurred;
    }

    private bool IsTetIntersectingBox(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 tetCenter, BoxCollider box, float margin)
    {
        Matrix4x4 worldToLocal = box.transform.worldToLocalMatrix;
        Vector3 halfSize = (box.size * 0.5f) + new Vector3(margin, margin, margin);
        Vector3 boxCenter = box.center;

        Vector3 lCenter = worldToLocal.MultiplyPoint3x4(tetCenter) - boxCenter;
        if (IsLocalPointInBox(lCenter, halfSize)) return true;

        Vector3 lp0 = worldToLocal.MultiplyPoint3x4(p0) - boxCenter;
        Vector3 lp1 = worldToLocal.MultiplyPoint3x4(p1) - boxCenter;
        Vector3 lp2 = worldToLocal.MultiplyPoint3x4(p2) - boxCenter;
        Vector3 lp3 = worldToLocal.MultiplyPoint3x4(p3) - boxCenter;

        int insideVerts = 0;
        if (IsLocalPointInBox(lp0, halfSize)) insideVerts++;
        if (IsLocalPointInBox(lp1, halfSize)) insideVerts++;
        if (IsLocalPointInBox(lp2, halfSize)) insideVerts++;
        if (IsLocalPointInBox(lp3, halfSize)) insideVerts++;

        if (insideVerts >= 1) return true;

        bool posZ = lp0.z > 0 || lp1.z > 0 || lp2.z > 0 || lp3.z > 0;
        bool negZ = lp0.z < 0 || lp1.z < 0 || lp2.z < 0 || lp3.z < 0;

        if (posZ && negZ)
        {
            if (Mathf.Abs(lCenter.x) <= halfSize.x && Mathf.Abs(lCenter.y) <= halfSize.y)
            {
                return true;
            }
        }

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

        bool oldHitBackfaces = Physics.queriesHitBackfaces;
        Physics.queriesHitBackfaces = true;

        Vector3 start = meshCol.bounds.min - new Vector3(0.1f, 0.1f, 0.1f);
        Vector3 dir = worldPoint - start;
        float dist = dir.magnitude;
        if (dist < 0.0001f)
        {
            Physics.queriesHitBackfaces = oldHitBackfaces;
            return false;
        }
        dir /= dist;

        int numHits = Physics.RaycastNonAlloc(start, dir, raycastHitBuffer, dist);
        int hitCount = 0;

        for (int i = 0; i < numHits; i++)
        {
            if (raycastHitBuffer[i].collider == meshCol)
            {
                hitCount++;
            }
        }

        Physics.queriesHitBackfaces = oldHitBackfaces;
        return (hitCount % 2) == 1;
    }

    public void PerformPlaneSweepCut(Vector3 startPt, Vector3 endPt, Vector3 sweepNorm, float sweepLength, Vector3 planeNormal)
    {
        activeBodies.RemoveAll(b => b == null);
        if (activeBodies.Count == 0) RefreshBodyCache();

        for (int bIdx = activeBodies.Count - 1; bIdx >= 0; bIdx--)
        {
            var body = activeBodies[bIdx];
            if (!CanCutBody(body)) continue;

            TetrahedralMesh tetMesh = body.TetMesh;
            Vector3[] positions = body.CurrentPositions;

            if (tetMesh == null || positions == null) continue;

            bool meshCutOccurred = false;

            for (int i = 0; i < tetMesh.tets.Count; i++)
            {
                var tet = tetMesh.tets[i];
                if (!tet.active) continue;

                if (tet.v0 >= positions.Length || tet.v1 >= positions.Length ||
                    tet.v2 >= positions.Length || tet.v3 >= positions.Length) continue;

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

                bool holdsPos = d0 > 0.001f || d1 > 0.001f || d2 > 0.001f || d3 > 0.001f;
                bool holdsNeg = d0 < -0.001f || d1 < -0.001f || d2 < -0.001f || d3 < -0.001f;

                if (holdsPos && holdsNeg)
                {
                    tet.active = false;
                    tetMesh.tets[i] = tet;
                    meshCutOccurred = true;
                }
            }

            if (meshCutOccurred)
            {
                RecordCutForBody(body);
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