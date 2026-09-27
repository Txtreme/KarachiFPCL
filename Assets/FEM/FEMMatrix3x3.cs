using UnityEngine;

[System.Serializable]
public struct FEMMatrix3x3
{
    public float m00, m01, m02;
    public float m10, m11, m12;
    public float m20, m21, m22;

    public FEMMatrix3x3(Vector3 col0, Vector3 col1, Vector3 col2)
    {
        m00 = col0.x; m10 = col0.y; m20 = col0.z;
        m01 = col1.x; m11 = col1.y; m21 = col1.z;
        m02 = col2.x; m12 = col2.y; m22 = col2.z;
    }

    public FEMMatrix3x3(
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
    {
        this.m00 = m00; this.m01 = m01; this.m02 = m02;
        this.m10 = m10; this.m11 = m11; this.m12 = m12;
        this.m20 = m20; this.m21 = m21; this.m22 = m22;
    }

    public static FEMMatrix3x3 Identity => new FEMMatrix3x3(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    );

    public float Determinant()
    {
        return m00 * (m11 * m22 - m12 * m21)
             - m01 * (m10 * m22 - m12 * m20)
             + m02 * (m10 * m21 - m11 * m20);
    }

    public FEMMatrix3x3 Inverse()
    {
        float det = Determinant();
        if (Mathf.Abs(det) < 1e-7f) return Identity;

        float invDet = 1.0f / det;
        return new FEMMatrix3x3(
            (m11 * m22 - m12 * m21) * invDet,
            (m02 * m21 - m01 * m22) * invDet,
            (m01 * m12 - m02 * m11) * invDet,

            (m12 * m20 - m10 * m22) * invDet,
            (m00 * m22 - m02 * m20) * invDet,
            (m02 * m10 - m00 * m12) * invDet,

            (m10 * m21 - m11 * m20) * invDet,
            (m01 * m20 - m00 * m21) * invDet,
            (m00 * m11 - m01 * m10) * invDet
        );
    }

    public Vector3 MultiplyVector(Vector3 v)
    {
        return new Vector3(
            m00 * v.x + m01 * v.y + m02 * v.z,
            m10 * v.x + m11 * v.y + m12 * v.z,
            m20 * v.x + m21 * v.y + m22 * v.z
        );
    }
}