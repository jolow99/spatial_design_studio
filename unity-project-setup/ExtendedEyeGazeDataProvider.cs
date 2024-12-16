using UnityEngine;
using System;
using System.Threading.Tasks;
using Microsoft.MixedReality.OpenXR;
using Microsoft.MixedReality.EyeTracking;

[DisallowMultipleComponent]
public class ExtendedEyeGazeDataProvider : MonoBehaviour
{
    public enum GazeType
    {
        Left,
        Right,
        Combined
    }

    public struct GazeReading
    {
        public bool IsValid;
        public Vector3 EyePosition;
        public Vector3 GazeDirection;
        public float Confidence;
        public DateTime Timestamp;

        public GazeReading(bool isValid, Vector3 position, Vector3 direction, float confidence = 1.0f)
        {
            IsValid = isValid;
            EyePosition = position;
            GazeDirection = direction;
            Confidence = confidence;
            Timestamp = DateTime.Now;
        }
    }

    [Header("Debug Settings")]
    [SerializeField] private bool showDebugLogs = false;
    [SerializeField] private bool visualizeGaze = false;

    [Header("Tracking Settings")]
    [SerializeField] private GazeType preferredGazeType = GazeType.Combined;
    [SerializeField] private float maxTimestampDelta = 0.1f; // Maximum gaze data interval

    private Camera _mainCamera;
    private EyeGazeTrackerWatcher _watcher;
    private EyeGazeTracker _eyeGazeTracker;
    private GazeReading _invalidGazeReading;
    private bool _gazePermissionEnabled;
    private SpatialGraphNode _eyeGazeTrackerNode;
    private Transform _mixedRealityPlayspace;
    private bool _isInitialized;
    private DateTime _lastValidReading = DateTime.MinValue;

    public bool IsEyeTrackingAvailable => _isInitialized && _eyeGazeTracker != null && _gazePermissionEnabled;
    public float LastConfidence { get; private set; }

    private void Awake()
    {
        _invalidGazeReading = new GazeReading(false, Vector3.zero, Vector3.zero, 0f);
        _isInitialized = false;
    }

    private async void Start()
    {
        await InitializeEyeTracking();
    }

    private async Task InitializeEyeTracking()
    {
        try
        {
            LogDebug("Initializing ExtendedEyeGazeDataProvider");
            
            _mainCamera = Camera.main;
            if (_mainCamera == null)
            {
                Debug.LogError("Main camera not found!");
                return;
            }

            _mixedRealityPlayspace = _mainCamera.transform.parent;

#if ENABLE_WINMD_SUPPORT
            LogDebug("Requesting eye tracking permissions...");
            _gazePermissionEnabled = await RequestEyeTrackingPermission();
#else
            
            _gazePermissionEnabled = true;
#endif

            if (!_gazePermissionEnabled)
            {
                Debug.LogError("Eye tracking permission denied or not available");
                return;
            }

            await InitializeEyeGazeTracker();
            _isInitialized = true;
            LogDebug("Eye tracking initialization complete");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize eye tracking: {ex.Message}\n{ex.StackTrace}");
            _isInitialized = false;
        }
    }

    private async Task InitializeEyeGazeTracker()
    {
        try
        {
            _watcher = new EyeGazeTrackerWatcher();
            _watcher.EyeGazeTrackerAdded += OnEyeGazeTrackerAdded;
            _watcher.EyeGazeTrackerRemoved += OnEyeGazeTrackerRemoved;
            await _watcher.StartAsync();
            LogDebug("Eye gaze tracker watcher started");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize eye gaze tracker: {ex.Message}");
            throw;
        }
    }

    private void OnEyeGazeTrackerRemoved(object sender, EyeGazeTracker tracker)
    {
        LogDebug("Eye gaze tracker removed");
        if (_eyeGazeTracker == tracker)
        {
            _eyeGazeTracker = null;
            _isInitialized = false;
        }
    }

    private async void OnEyeGazeTrackerAdded(object sender, EyeGazeTracker tracker)
    {
        LogDebug("Eye gaze tracker added");
        try
        {
            await tracker.OpenAsync(true);
            _eyeGazeTracker = tracker;
            
            var supportedFrameRates = _eyeGazeTracker.SupportedTargetFrameRates;
            LogDebug($"Available frame rates: {string.Join(", ", supportedFrameRates)}");

            // Set to highest available frame rate, 90Hz
            if (supportedFrameRates.Count > 0)
            {
                var highestFrameRate = supportedFrameRates[supportedFrameRates.Count - 1];
                _eyeGazeTracker.SetTargetFrameRate(highestFrameRate);
                LogDebug($"Set frame rate to: {highestFrameRate.FramesPerSecond} FPS");
            }

            _eyeGazeTrackerNode = SpatialGraphNode.FromDynamicNodeId(tracker.TrackerSpaceLocatorNodeId);
            LogDebug("Eye gaze tracker successfully initialized");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to open eye gaze tracker: {ex.Message}\n{ex.StackTrace}");
            _isInitialized = false;
        }
    }

    public GazeReading GetWorldSpaceGazeReading()
    {
        return GetWorldSpaceGazeReading(preferredGazeType, DateTime.Now);
    }

    public GazeReading GetWorldSpaceGazeReading(GazeType gazeType, DateTime timestamp)
    {
        if (!IsEyeTrackingAvailable || _eyeGazeTracker == null)
        {
            return _invalidGazeReading;
        }

        var reading = _eyeGazeTracker.TryGetReadingAtTimestamp(timestamp);
        if (reading == null)
        {
            LogDebug($"No eye tracking reading available at timestamp: {timestamp}");
            return _invalidGazeReading;
        }

        // Check if reading is too old
        var age = (DateTime.Now - timestamp).TotalSeconds;
        if (age > maxTimestampDelta)
        {
            LogDebug($"Eye tracking reading too old: {age:F3}s");
            return _invalidGazeReading;
        }

        System.Numerics.Vector3 origin = System.Numerics.Vector3.Zero;
        System.Numerics.Vector3 direction = System.Numerics.Vector3.Zero;
        bool success = false;
        float confidence = 1.0f; 

        switch (gazeType)
        {
            case GazeType.Left:
                success = reading.TryGetLeftEyeGazeInTrackerSpace(out origin, out direction);
                break;
            case GazeType.Right:
                success = reading.TryGetRightEyeGazeInTrackerSpace(out origin, out direction);
                break;
            case GazeType.Combined:
                success = reading.TryGetCombinedEyeGazeInTrackerSpace(out origin, out direction);
                break;
        }

        if (!success)
        {
            LogDebug($"Invalid gaze reading - Success: {success}");
            return _invalidGazeReading;
        }

        if (!_eyeGazeTrackerNode.TryLocate(reading.SystemRelativeTime.Ticks, out Pose eyeGazeTrackerPose))
        {
            LogDebug("Failed to locate eye gaze tracker node");
            return _invalidGazeReading;
        }

        Matrix4x4 eyeGazeTrackerSpaceToWorld = (_mixedRealityPlayspace != null) ?
            _mixedRealityPlayspace.localToWorldMatrix * Matrix4x4.TRS(eyeGazeTrackerPose.position, eyeGazeTrackerPose.rotation, Vector3.one) :
            Matrix4x4.TRS(eyeGazeTrackerPose.position, eyeGazeTrackerPose.rotation, Vector3.one);

        Vector3 eyePosition = eyeGazeTrackerSpaceToWorld.MultiplyPoint3x4(new Vector3(origin.X, origin.Y, -origin.Z));
        Vector3 gazeDirection = eyeGazeTrackerSpaceToWorld.MultiplyVector(new Vector3(direction.X, direction.Y, -direction.Z)).normalized;

        LastConfidence = confidence;
        _lastValidReading = timestamp;

        if (visualizeGaze)
        {
            Debug.DrawRay(eyePosition, gazeDirection * 10f, Color.green, 0.016f);
        }

        return new GazeReading(true, eyePosition, gazeDirection, confidence);
    }

#if ENABLE_WINMD_SUPPORT
    private async Task<bool> RequestEyeTrackingPermission()
    {
        try
        {
            var accessStatus = await Windows.Perception.People.EyesPose.RequestAccessAsync();
            LogDebug($"Eye tracking permission status: {accessStatus}");
            return accessStatus == Windows.UI.Input.GazeInputAccessStatus.Allowed;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to request eye tracking permission: {ex.Message}");
            return false;
        }
    }
#endif

    private void LogDebug(string message)
    {
        if (showDebugLogs)
        {
            Debug.Log($"[EyeGazeProvider] {message}");
        }
    }

    private void OnDisable()
    {
        if (_watcher != null)
        {
            _watcher.EyeGazeTrackerAdded -= OnEyeGazeTrackerAdded;
            _watcher.EyeGazeTrackerRemoved -= OnEyeGazeTrackerRemoved;
        }
    }

    private void OnDestroy()
    {
        _eyeGazeTracker = null;
        _watcher = null;
    }
}