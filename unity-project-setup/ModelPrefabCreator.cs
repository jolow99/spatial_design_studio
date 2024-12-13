#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Linq;
using System.Collections.Generic;

public class ModelPrefabCreator : EditorWindow
{
    private Object sourceFolder;
    private string prefabPath = "Assets/Prefabs";
    private Vector3 defaultScale = Vector3.one;
    private Vector3 defaultRotation = Vector3.zero;
    
    [MenuItem("Tools/Bulk Model Prefab Creator")]
    public static void ShowWindow()
    {
        GetWindow<ModelPrefabCreator>("Prefab Creator");
    }

    private void OnGUI()
    {
        GUILayout.Label("Bulk Create Prefabs", EditorStyles.boldLabel);
        
        sourceFolder = EditorGUILayout.ObjectField("Models Folder", sourceFolder, typeof(Object), false);
        prefabPath = EditorGUILayout.TextField("Prefab Output Path", prefabPath);
        
        defaultScale = EditorGUILayout.Vector3Field("Default Scale", defaultScale);
        defaultRotation = EditorGUILayout.Vector3Field("Default Rotation", defaultRotation);

        if (GUILayout.Button("Create Prefabs"))
        {
            CreatePrefabs();
        }
        
        if (GUILayout.Button("Setup ExperimentManager"))
        {
            SetupExperimentManager();
        }
    }

    private void CreatePrefabs()
    {
        if (sourceFolder == null) return;

        string sourcePath = AssetDatabase.GetAssetPath(sourceFolder);
        string[] modelFiles = Directory.GetFiles(sourcePath, "*.*", SearchOption.TopDirectoryOnly)
            .Where(file => file.EndsWith(".fbx") || file.EndsWith(".obj"))
            .ToArray();

        if (!Directory.Exists(prefabPath))
        {
            Directory.CreateDirectory(prefabPath);
        }

        foreach (string modelPath in modelFiles)
        {
            GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
            if (model == null) continue;

            // Create instance and configure
            GameObject instance = PrefabUtility.InstantiatePrefab(model) as GameObject;
            if (instance == null) continue;

            instance.transform.localScale = defaultScale;
            instance.transform.rotation = Quaternion.Euler(defaultRotation);

            // Create prefab
            string prefabName = Path.GetFileNameWithoutExtension(modelPath);
            string fullPrefabPath = $"{prefabPath}/{prefabName}_Prefab.prefab";
            
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(instance, fullPrefabPath);
            if (prefab == null)
            {
                Debug.LogError($"Failed to create prefab for {prefabName}");
            }
            
            DestroyImmediate(instance);
        }
        
        AssetDatabase.Refresh();
        Debug.Log("Finished creating prefabs!");
    }

    private void SetupExperimentManager()
    {
        // Find ExperimentManager in scene
        ExperimentManager manager = FindObjectOfType<ExperimentManager>();
        if (manager == null)
        {
            Debug.LogError("ExperimentManager not found in scene!");
            return;
        }

        // Get all prefabs
        string[] prefabGuids = AssetDatabase.FindAssets("t:Prefab", new[] { prefabPath });
        List<GameObject> modelList = new List<GameObject>();

        foreach (string guid in prefabGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            
            if (prefab != null)
            {
                modelList.Add(prefab);
            }
        }

        // Apply to manager
        SerializedObject serializedManager = new SerializedObject(manager);
        
        // Set the first prefab as practice prefab
        if (modelList.Count > 0)
        {
            SerializedProperty practicePrefabProp = serializedManager.FindProperty("practicePrefab");
            practicePrefabProp.objectReferenceValue = modelList[0];
            modelList.RemoveAt(0); // Remove it from the list so it's not used in experiment prefabs
        }

        // Set the rest as experiment prefabs
        SerializedProperty experimentPrefabsProp = serializedManager.FindProperty("experimentPrefabs");
        experimentPrefabsProp.ClearArray();
        
        for (int i = 0; i < modelList.Count; i++)
        {
            experimentPrefabsProp.InsertArrayElementAtIndex(i);
            SerializedProperty element = experimentPrefabsProp.GetArrayElementAtIndex(i);
            element.objectReferenceValue = modelList[i];
        }
        
        serializedManager.ApplyModifiedProperties();
        Debug.Log($"Successfully set up ExperimentManager with {modelList.Count} experiment models!");
    }
}
#endif