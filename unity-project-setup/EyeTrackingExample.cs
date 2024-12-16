using UnityEngine;
using TMPro;
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using MixedReality.Toolkit;
#if WINDOWS_UWP
using Windows.Storage;
#endif

public class EyeTrackingExample : MonoBehaviour
{
    [Header("References")]
    public TextMeshPro debugTextMesh;
    public Transform targetObject;

    [Header("Settings")]
    public float maxRaycastDistance = 10f;

    [Header("Gaze Settings")]
    [SerializeField] private float updateInterval = 0.0111f; // 90Hz

    private ExtendedEyeGazeDataProvider extendedEyeGazeDataProvider;
    private string gazeFilePath;
    private string sessionTimestamp;
    private float lastUpdateTime;
    private RaycastHit raycastHit;
    private readonly Dictionary<string, bool> ignoredObjectsCache = new Dictionary<string, bool>();
    private GameObject currentGazeObject;
    private bool isInitialized = false;
    private StringBuilder gazeDataToWrite = new StringBuilder();
    private Vector3 currentGazePoint;
    private bool hasValidGaze;
    private int currentObjectNumber;
    private string currentObjectName;

    private void Start()
    {
        // Set once to exactly 90Hz
        Time.fixedDeltaTime = 1f/90f;
        QualitySettings.vSyncCount = 0;  // Disable VSync
        Application.targetFrameRate = 90;

        sessionTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        InitializeComponents();
        InitializeLogging();
        CacheIgnoredObjects();
        
        if (debugTextMesh != null)
        {
            debugTextMesh.gameObject.SetActive(true);
            debugTextMesh.text = "Eye tracking initialized...";
        }
        isInitialized = true;
    }

    private void InitializeComponents()
    {
        extendedEyeGazeDataProvider = FindObjectOfType<ExtendedEyeGazeDataProvider>();
        if (extendedEyeGazeDataProvider == null)
        {
            var dataProvider = new GameObject("EyeGazeDataProvider");
            extendedEyeGazeDataProvider = dataProvider.AddComponent<ExtendedEyeGazeDataProvider>();
        }
    }

    public void ResetGazeData(int objectNumber, string objectName) // Modified to accept object name
    {
        currentObjectNumber = objectNumber;
        currentObjectName = objectName.Replace(" ", "_"); // Replace spaces with underscores
        gazeDataToWrite.Clear();
        sessionTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        InitializeLogging();
    }

    private void InitializeLogging()
    {
        try
        {
            string folder = 
#if WINDOWS_UWP
                ApplicationData.Current.LocalFolder.Path;
#else
                Application.persistentDataPath;
#endif
            
            gazeFilePath = Path.Combine(folder, $"eyetrackingdata__{currentObjectName}.csv"); // save eyetracking data for each model
            Directory.CreateDirectory(Path.GetDirectoryName(gazeFilePath));

            string header = "Timestamp,HitObject,HitPointX,HitPointY,HitPointZ,NormalX,NormalY,NormalZ\n";
            
            File.WriteAllText(gazeFilePath, header);
            Debug.Log($"Created log file at: {gazeFilePath}"); // save data
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize logging: {ex}");
        }
    }

    private void CacheIgnoredObjects()
    {
        ignoredObjectsCache.Clear();
        foreach (var name in ignoreObjectNames)
        {
            ignoredObjectsCache[name] = true;
        }
    }

    private void FixedUpdate()
    {
        if (!isInitialized || extendedEyeGazeDataProvider == null) return;
        UpdateGazeTracking();
    }
    
    private void UpdateGazeTracking()
    {
        var gazeReading = extendedEyeGazeDataProvider.GetWorldSpaceGazeReading();
        
        hasValidGaze = Physics.Raycast(gazeReading.EyePosition, gazeReading.GazeDirection, out raycastHit, maxRaycastDistance);
        
        if (hasValidGaze && !ignoredObjectsCache.ContainsKey(raycastHit.collider.gameObject.name))
        {
            currentGazePoint = raycastHit.point;
            currentGazeObject = raycastHit.collider.gameObject;
            
            // Get the mesh transform reference if it exists
            var meshTransformRef = targetObject.GetComponent<MeshTransformReference>();
            
            // to maintain recording of coords in rel. local space of object
            if (meshTransformRef != null)
            {
                // First transform to object space
                Vector3 objectSpacePoint = targetObject.InverseTransformPoint(currentGazePoint);
                Vector3 objectSpaceNormal = targetObject.InverseTransformDirection(raycastHit.normal);
                
                // Then apply the original mesh-to-object transformation
                Matrix4x4 originalTransform = meshTransformRef.OriginalMeshToObject;
                Vector3 localHitPoint = originalTransform.MultiplyPoint3x4(objectSpacePoint);
                Vector3 localNormal = originalTransform.MultiplyVector(objectSpaceNormal);
                
                LogGazePoint(localHitPoint, localNormal, currentGazeObject.name);
                
                if (debugTextMesh != null)
                {
                    debugTextMesh.text = $"Looking at: {currentGazeObject.name}\n" +
                                    $"Local Point: {localHitPoint:F2}";
                }
            }
            else
            {
                // Fallback to original behavior if no reference exists
                Vector3 localHitPoint = targetObject.InverseTransformPoint(currentGazePoint);
                Vector3 localNormal = targetObject.InverseTransformDirection(raycastHit.normal);
                LogGazePoint(localHitPoint, localNormal, currentGazeObject.name);
            }
        }
        else
        {
            Vector3 worldEndPoint = gazeReading.EyePosition + gazeReading.GazeDirection * maxRaycastDistance;
            Vector3 localEndPoint = targetObject.InverseTransformPoint(worldEndPoint);
            
            LogGazePoint(localEndPoint, Vector3.zero, "None");
            
            if (debugTextMesh != null)
            {
                debugTextMesh.text = "Looking at: Nothing";
            }
        }

        if (Time.frameCount % 90 == 0)
        {
            WriteGazeDataToFile();
        }
    }

    private void LogGazePoint(Vector3 localPosition, Vector3 localNormal, string objectName)
    {
        string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}," +
                         $"{objectName}," +
                         $"{localPosition.x:F4},{localPosition.y:F4},{localPosition.z:F4}," +
                         $"{localNormal.x:F4},{localNormal.y:F4},{localNormal.z:F4}\n";
        
        gazeDataToWrite.Append(logEntry);
    }

    private void WriteGazeDataToFile()
    {
        if (gazeDataToWrite.Length == 0) return;

        try
        {
#if WINDOWS_UWP
            UnityEngine.WSA.Application.InvokeOnUIThread(async () =>
            {
                try
                {
                    StorageFolder folder = ApplicationData.Current.LocalFolder;
                    StorageFile file = await folder.GetFileAsync(Path.GetFileName(gazeFilePath));
                    await FileIO.AppendTextAsync(file, gazeDataToWrite.ToString());
                    gazeDataToWrite.Clear();
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Failed to write gaze data on UWP: {ex}");
                }
            }, false);
#else
            File.AppendAllText(gazeFilePath, gazeDataToWrite.ToString());
            gazeDataToWrite.Clear();
#endif
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to write gaze data: {ex}");
        }
    }

    private void OnDisable()
    {
        WriteGazeDataToFile();
    }

    private void OnApplicationQuit() // save file on applciation quit/next model
    {
        WriteGazeDataToFile();
    }
}