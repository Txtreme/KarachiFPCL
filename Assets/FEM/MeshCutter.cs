using UnityEngine;

public class MeshCutter : MonoBehaviour
{
    [Header("Cutter Extents")]
    [Tooltip("Maximum radius around the cutter blade that can slice tetrahedra.")]
    public float cutRadius = 0.5f;

    private Vector3 prevPosition;

    private void Start()
    {
        prevPosition = transform.position;
    }

    private void Update()
    {
        Vector3 currentPosition = transform.position;
        Vector3 sweepDir = currentPosition - prevPosition;
        float sweepLength = sweepDir.magnitude;

        if (sweepLength > 0.001f)
        {
            Vector3 sweepNorm = sweepDir / sweepLength;

            // Cutting plane normal orthogonal to sweep direction
            Vector3 planeNormal = Vector3.Cross(sweepNorm, transform.forward).normalized;
            if (planeNormal == Vector3.zero) planeNormal = transform.up;

            PerformCut(prevPosition, currentPosition, sweepNorm, sweepLength, planeNormal);
        }

        prevPosition = currentPosition;
    }

    public void PerformCut(Vector3 startPt, Vector3 endPt, Vector3 sweepNorm, float sweepLength, Vector3 planeNormal)
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

                // 1. Distance check to the cutter sweep line segment
                float proj = Vector3.Dot(tetCenter - startPt, sweepNorm);
                proj = Mathf.Clamp(proj, 0f, sweepLength);
                Vector3 closestPointOnSweep = startPt + sweepNorm * proj;

                if (Vector3.Distance(tetCenter, closestPointOnSweep) > cutRadius)
                {
                    continue; // Skip tetrahedra outside blade range
                }

                // 2. Signed distances to cutting plane
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
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, cutRadius);
    }
}