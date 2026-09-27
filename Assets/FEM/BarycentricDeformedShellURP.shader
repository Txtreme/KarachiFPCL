Shader "FEM/BarycentricDeformedShellURP"
{
    Properties
    {
        [MainTexture] _BaseMap ("Base Map", 2D) = "white" {}
        [MainColor] _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _Smoothness ("Smoothness", Range(0.0, 1.0)) = 0.5
        _Metallic ("Metallic", Range(0.0, 1.0)) = 0.0
    }

    SubShader
    {
        Tags 
        { 
            "RenderType" = "Opaque" 
            "RenderPipeline" = "UniversalPipeline" 
            "Queue" = "Geometry"
            "DisableBatching" = "True"
        }
        LOD 300
        Cull Off

        // ------------------------------------------------------------------
        // Forward Lit Pass
        // ------------------------------------------------------------------
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float3 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float2 uv : TEXCOORD2;
            };

            StructuredBuffer<float3> _DeformedVertices;
            StructuredBuffer<float3> _DeformedNormals;

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                half4 _BaseColor;
                half _Smoothness;
                half _Metallic;
            CBUFFER_END

            Texture2D _BaseMap;
            SamplerState sampler_BaseMap;

            float3 SafeNormalize(float3 v, float3 fallback)
            {
                float lenSq = dot(v, v);
                return lenSq > 1e-6f ? v * rsqrt(lenSq) : fallback;
            }

            Varyings vert(Attributes input)
            {
                Varyings output;

                float3 posWS = _DeformedVertices[input.vertexID];
                float3 rawNorm = _DeformedNormals[input.vertexID];

                // Fallback to standard mesh transform if buffer is unpopulated or zeroed
                if (dot(posWS, posWS) < 1e-6f)
                {
                    posWS = TransformObjectToWorld(input.positionOS);
                    rawNorm = TransformObjectToWorldNormal(input.normalOS);
                }

                output.positionWS = posWS;
                output.positionCS = TransformWorldToHClip(posWS);
                output.normalWS = SafeNormalize(rawNorm, float3(0.0, 1.0, 0.0));
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);

                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                half4 albedoMap = _BaseMap.Sample(sampler_BaseMap, input.uv) * _BaseColor;
                
                SurfaceData surfaceData = (SurfaceData)0;
                surfaceData.albedo = albedoMap.rgb;
                surfaceData.alpha = albedoMap.a;
                surfaceData.metallic = _Metallic;
                surfaceData.smoothness = _Smoothness;
                surfaceData.normalTS = half3(0.0, 0.0, 1.0);
                surfaceData.occlusion = 1.0;

                InputData inputData = (InputData)0;
                inputData.positionWS = input.positionWS;
                inputData.normalWS = SafeNormalize(input.normalWS, float3(0.0, 1.0, 0.0));
                inputData.viewDirectionWS = GetWorldSpaceNormalizeViewDir(input.positionWS);
                inputData.shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                inputData.bakedGI = SampleSH(inputData.normalWS);

                return UniversalFragmentPBR(inputData, surfaceData);
            }
            ENDHLSL
        }

        // ------------------------------------------------------------------
        // Shadow Caster Pass
        // ------------------------------------------------------------------
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vertShadow
            #pragma fragment fragShadow

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float3 positionOS : POSITION;
                float3 normalOS : NORMAL;
                uint vertexID : SV_VertexID;
            };

            struct VaryingsShadow
            {
                float4 positionCS : SV_POSITION;
            };

            StructuredBuffer<float3> _DeformedVertices;
            StructuredBuffer<float3> _DeformedNormals;

            float3 _LightDirection;

            float4 GetShadowPositionHClip(float3 positionWS, float3 normalWS)
            {
                float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, _LightDirection));
                #if UNITY_REVERSED_Z
                    positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #else
                    positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #endif
                return positionCS;
            }

            VaryingsShadow vertShadow(Attributes input)
            {
                VaryingsShadow output;
                float3 posWS = _DeformedVertices[input.vertexID];
                float3 rawNorm = _DeformedNormals[input.vertexID];

                if (dot(posWS, posWS) < 1e-6f)
                {
                    posWS = TransformObjectToWorld(input.positionOS);
                    rawNorm = TransformObjectToWorldNormal(input.normalOS);
                }

                float lenSq = dot(rawNorm, rawNorm);
                float3 normWS = lenSq > 1e-6f ? rawNorm * rsqrt(lenSq) : float3(0.0, 1.0, 0.0);

                output.positionCS = GetShadowPositionHClip(posWS, normWS);
                return output;
            }

            half4 fragShadow(VaryingsShadow input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }

        // ------------------------------------------------------------------
        // Depth-Only Pass
        // ------------------------------------------------------------------
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask 0

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vertDepth
            #pragma fragment fragDepth

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float3 positionOS : POSITION;
                uint vertexID : SV_VertexID;
            };

            struct VaryingsDepth
            {
                float4 positionCS : SV_POSITION;
            };

            StructuredBuffer<float3> _DeformedVertices;

            VaryingsDepth vertDepth(Attributes input)
            {
                VaryingsDepth output;
                float3 posWS = _DeformedVertices[input.vertexID];

                if (dot(posWS, posWS) < 1e-6f)
                {
                    posWS = TransformObjectToWorld(input.positionOS);
                }

                output.positionCS = TransformWorldToHClip(posWS);
                return output;
            }

            half4 fragDepth(VaryingsDepth input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}