using UnityEngine;
using System.IO;
using System.Text;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class BatchPointCloudExporter : MonoBehaviour
{
    [Header("Export Settings")]
    [Tooltip("Total number of points to generate per mesh")]
    public int pointsPerMesh = 15000;
    
    [Tooltip("Path to folder containing meshes to process")]
    public string meshFolderPath = "Assets/Models/ExperimentObjects";
    
    [Tooltip("Output folder for CSVs")]
    public string outputFolderPath = "Assets/Models/PointClouds";

#if UNITY_EDITOR
    [ContextMenu("Batch Export All Meshes")]
    private void BatchExportMeshes()
    {
        // Ensure output directory exists
        if (!Directory.Exists(outputFolderPath))
        {
            Directory.CreateDirectory(outputFolderPath);
        }

        // Get all mesh assets in the specified folder
        string[] meshGuids = AssetDatabase.FindAssets("t:Mesh", new[] { meshFolderPath });
        Debug.Log($"Found {meshGuids.Length} meshes to process");

        foreach (string guid in meshGuids)
        {
            string assetPath = AssetDatabase.GUIDToAssetPath(guid);
            Mesh mesh = AssetDatabase.LoadAssetAtPath<Mesh>(assetPath);
            
            if (mesh != null)
            {
                ExportMeshPoints(mesh, Path.GetFileNameWithoutExtension(assetPath));
            }
        }
        
        AssetDatabase.Refresh();
        Debug.Log("Batch export complete!");
    }

    private void ExportMeshPoints(Mesh mesh, string meshName)
    {
        try
        {
            List<Vector3> sampledPoints = new List<Vector3>();
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;

            // Calculate total surface area and collect triangle data
            float totalArea = 0f;
            List<(Vector3 v1, Vector3 v2, Vector3 v3, float area)> triangleData = new List<(Vector3, Vector3, Vector3, float)>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                Vector3 v1 = vertices[triangles[i]];
                Vector3 v2 = vertices[triangles[i + 1]];
                Vector3 v3 = vertices[triangles[i + 2]];

                float area = Vector3.Cross(v2 - v1, v3 - v1).magnitude * 0.5f;
                if (area > 0.0001f) // Filter out degenerate triangles
                {
                    triangleData.Add((v1, v2, v3, area));
                    totalArea += area;
                }
            }

            // Sort triangles by area (largest first) for better distribution
            triangleData.Sort((a, b) => b.area.CompareTo(a.area));

            // Calculate cumulative areas
            float[] cumulativeAreas = new float[triangleData.Count];
            float currentSum = 0f;
            for (int i = 0; i < triangleData.Count; i++)
            {
                currentSum += triangleData[i].area;
                cumulativeAreas[i] = currentSum;
            }

            // Generate points
            for (int i = 0; i < pointsPerMesh; i++)
            {
                float randomArea = Random.Range(0f, totalArea); // Random per triangle
                int triangleIndex = System.Array.BinarySearch(cumulativeAreas, randomArea);
                if (triangleIndex < 0)
                {
                    triangleIndex = ~triangleIndex;
                }
                
                var triangle = triangleData[triangleIndex];

                float u = Random.Range(0f, 1f);
                float v = Random.Range(0f, 1f);
                
                if (u + v > 1f)
                {
                    u = 1f - u;
                    v = 1f - v;
                }

                float w = 1f - u - v;

                Vector3 point = triangle.v1 * u + triangle.v2 * v + triangle.v3 * w;
                sampledPoints.Add(point);
            }

            // Create CSV mesh file
            StringBuilder csv = new StringBuilder();
            csv.AppendLine("x,y,z");
            foreach (Vector3 point in sampledPoints)
            {
                csv.AppendLine($"{point.x:F4},{point.y:F4},{point.z:F4}");
            }

            // Save to output folder
            string filePath = Path.Combine(outputFolderPath, $"{meshName}_points.csv");
            File.WriteAllText(filePath, csv.ToString());
            
            Debug.Log($"Exported {sampledPoints.Count} points for {meshName} to: {filePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to export points for {meshName}: {e.Message}");
        }
    }
#endif

    public void Initialize(string timestamp)
    {
        string folderPath = Path.Combine(Application.persistentDataPath, "PointCloudExports");
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }
    }
}