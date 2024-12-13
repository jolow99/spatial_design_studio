using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using MixedReality.Toolkit;

public class ExperimentManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform spawnPoint;

    [Header("UI References")]
    [SerializeField] private StatefulInteractable readyButton;
    [SerializeField] private TextMeshPro instructionText;
    [SerializeField] private TextMeshPro countdownText;

    [Header("Audio References")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip beepSound;

    [Header("Experiment Objects")]
    [SerializeField] private GameObject practicePrefab;
    [SerializeField] private GameObject[] experimentPrefabs;
    
    [Header("Settings")]
    [SerializeField] private float countdownTime = 5f; // initial countdown timer
    [SerializeField] private float modelViewTime = 75f; // 75 sec
    [SerializeField] private Vector3 initialScale = new Vector3(0.1f, 0.1f, 0.1f); 
    
    private GameObject currentObject;
    private int currentExperimentIndex = -1;
    private EyeTrackingExample eyeTrackingController;
    private bool isExperimentActive = false;
    private float currentModelTimer = 0f;
    private bool isCountingDown = false;

    private void Start()
    {
        // Initialize UI button
        if (readyButton != null)
            readyButton.OnClicked.AddListener(OnReadyButtonPressed);
        
        // Get eye tracking controller
        eyeTrackingController = GetComponent<EyeTrackingExample>();
        if (eyeTrackingController == null)
        {
            eyeTrackingController = gameObject.AddComponent<EyeTrackingExample>();
        }

        // Verify audio components
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.playOnAwake = false;
        }

        // Hide countdown text initially
        if (countdownText != null)
        {
            countdownText.gameObject.SetActive(false);
        }
        
        // Start with practice object
        SpawnPracticeObject();
        
        UpdateInstructionText("Please familiarize yourself with the controls.\nPress 'Ready' when you want to begin the experiment.");
    }

    private void Update()
    {
        if (isExperimentActive)
        {
            currentModelTimer += Time.deltaTime;
            float timeLeft = modelViewTime - currentModelTimer;

            // Handle countdown sounds for last 5 seconds
            if (timeLeft <= 5f && !isCountingDown)
            {
                isCountingDown = true;
                StartCoroutine(PlayCountdownBeeps());
            }

            // Check if it's time to switch models
            if (currentModelTimer >= modelViewTime)
            {
                currentModelTimer = 0f;
                isCountingDown = false;
                MoveToNextModel();
            }
        }
    }

    private IEnumerator PlayCountdownBeeps()
    {
        // Play beep once per second for the last 5 seconds (audio file)
        for (int i = 0; i < 5; i++)
        {
            audioSource.PlayOneShot(beepSound);
            yield return new WaitForSeconds(1.0f);
        }
    }

    private void SpawnPracticeObject()
    {
        if (currentObject != null) Destroy(currentObject);
        currentObject = Instantiate(practicePrefab, spawnPoint.position, spawnPoint.rotation);
        SetupObjectManipulation(currentObject);
        eyeTrackingController.enabled = false;
    }

    private void SpawnExperimentObject(int index)
    {
        if (currentObject != null) Destroy(currentObject);
        currentObject = Instantiate(experimentPrefabs[index], spawnPoint.position, spawnPoint.rotation);
        SetupObjectManipulation(currentObject);
        eyeTrackingController.targetObject = currentObject.transform;
        
        // Get the object name from the prefab and pass it along with the index
        string objectName = experimentPrefabs[index].name;
        eyeTrackingController.ResetGazeData(index, objectName);
        
        eyeTrackingController.enabled = true;
    }

    private void SetupObjectManipulation(GameObject obj)
    {
        obj.transform.localScale = initialScale;
        obj.transform.position = spawnPoint.position;
        obj.transform.rotation = spawnPoint.rotation;

        if (obj.TryGetComponent<MeshFilter>(out var meshFilter))
        {
            var meshTransformRef = obj.AddComponent<MeshTransformReference>();
            meshTransformRef.Initialize(meshFilter.transform);
        }
    }

    private void OnReadyButtonPressed()
    {
        // Hide the ready button, practice object, and instructions
        if (readyButton != null)
        {
            readyButton.gameObject.SetActive(false);
        }
        if (currentObject != null)
        {
            currentObject.SetActive(false);
        }
        if (instructionText != null)
        {
            instructionText.gameObject.SetActive(false);
        }

        // Show countdown text
        if (countdownText != null)
        {
            countdownText.gameObject.SetActive(true);
        }

        StartCoroutine(StartCountdown());
    }

    private IEnumerator StartCountdown()
    {
        // Initial countdown with visual timer
        float timeLeft = countdownTime;
        while (timeLeft > 0)
        {
            if (countdownText != null)
                countdownText.text = $"Starting in: {timeLeft:F0}";
            timeLeft -= Time.deltaTime;
            yield return null;
        }

        // Hide countdown text
        if (countdownText != null)
        {
            countdownText.gameObject.SetActive(false);
        }

        // Destroy the practice object completely before moving to experiment formal
        if (currentObject != null)
        {
            Destroy(currentObject);
        }

        // Move to first experiment object
        currentExperimentIndex = 0;
        currentModelTimer = 0f;
        isCountingDown = false;
        SpawnExperimentObject(currentExperimentIndex);
        isExperimentActive = true;
    }

    private void MoveToNextModel()
    {
        // Save current eye tracking data
        eyeTrackingController.enabled = false;

        currentExperimentIndex++;
        if (currentExperimentIndex >= experimentPrefabs.Length)
        {
            // Experiment complete
            EndExperiment();
        }
        else
        {
            // Move to next object without showing which number it is
            SpawnExperimentObject(currentExperimentIndex);
        }
    }

    private void EndExperiment()
    {
        isExperimentActive = false;
        if (currentObject != null) 
            Destroy(currentObject);
        
        // Show completion message
        if (instructionText != null)
        {
            instructionText.gameObject.SetActive(true);
            instructionText.text = "Experiment complete.\nThank you for participating!";
        }
    }

    private void UpdateInstructionText(string message)
    {
        if (instructionText != null && instructionText.gameObject.activeSelf)
        {
            instructionText.text = message;
        }
    }

    private void OnDisable()
    {
        // Clean up event listener in unity proj
        if (readyButton != null)
            readyButton.OnClicked.RemoveListener(OnReadyButtonPressed);
    }
}

public class MeshTransformReference : MonoBehaviour
{
    public Matrix4x4 OriginalMeshToObject { get; private set; }

    public void Initialize(Transform meshTransform)
    {
        // Store the original transformation from mesh to object space (maintains relative local coords)
        OriginalMeshToObject = transform.worldToLocalMatrix * meshTransform.localToWorldMatrix;
    }
}