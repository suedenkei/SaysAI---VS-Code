package com.example.saysai

import android.content.Context
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import org.json.JSONObject
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.common.FileUtil

data class Prediction(
    val index: Int,
    val label: String,
    val confidence: Float
)

class SignClassifier(
    private val context: Context,
    private val onReadyChanged: (Boolean) -> Unit = {}
) {
    companion object {
        private const val MODEL_PATH = "models/sign_model.tflite"
        private const val LABEL_MAP_PATH = "models/label_map.json"
    }

    private var interpreter: InterpreterApi? = null
    private var labels: List<String> = emptyList()

    var ready: Boolean = false
        private set

    init { load() }

    private fun load() {
        labels = loadLabels()
        val modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH)

        TfLite.initialize(context)
            .addOnSuccessListener {
                val options = InterpreterApi.Options()
                    .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                interpreter = InterpreterApi.create(modelBuffer, options)
                ready = true
                onReadyChanged(true)
            }
            .addOnFailureListener { e ->
                Log.e("SaysAI", "TfLite init failed", e)
                ready = false
                onReadyChanged(false)
            }
    }

    private fun loadLabels(): List<String> {
        val text = context.assets.open(LABEL_MAP_PATH).bufferedReader().use { it.readText() }
        val json = JSONObject(text)
        val indexToLabel = json.getJSONObject("index_to_label")
        val keys = indexToLabel.keys().asSequence()
            .map { it.toInt() }
            .sorted()
            .toList()
        return keys.map { indexToLabel.getString(it.toString()) }
    }

    fun predict(sequence: Array<FloatArray>): Prediction? {
        val localInterpreter = interpreter ?: return null
        if (!ready || sequence.size != 30) return null

        val input = arrayOf(sequence)
        val output = Array(1) { FloatArray(labels.size) }
        localInterpreter.run(input, output)

        val probs = output[0]
        var bestIndex = 0
        var bestScore = probs[0]

        for (i in 1 until probs.size) {
            if (probs[i] > bestScore) {
                bestScore = probs[i]
                bestIndex = i
            }
        }

        return Prediction(index = bestIndex, label = labels[bestIndex], confidence = bestScore)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}