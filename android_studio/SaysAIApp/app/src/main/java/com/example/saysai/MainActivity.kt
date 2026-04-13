package com.example.saysai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Switch
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.ArrayDeque
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), HandLandmarkerHelper.Listener {

    private lateinit var previewView: PreviewView
    private lateinit var resultText: TextView
    private lateinit var chipGroup: ChipGroup
    private lateinit var clearButton: Button
    private lateinit var doneButton: Button
    private lateinit var navHome: ImageView
    private lateinit var navCamera: ImageView
    private lateinit var navArchive: ImageView
    private lateinit var switchLeftHandMode: Switch

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var handLandmarkerHelper: HandLandmarkerHelper
    private lateinit var signClassifier: SignClassifier

    private val sequenceBuffer = ArrayDeque<FloatArray>()
    private var lastAcceptedMs = 0L
    private var lastDetectRequestMs = 0L

    private val collectedSigns = mutableListOf<String>()
    private val collectedAccuracies = mutableListOf<Int>()
    private val collectedResponseTimes = mutableListOf<Long>()
    private var lastAddedLabel = ""
    private var lastAddedMs = 0L
    private var isStarted = false
    private var noHandSinceMs = 0L
    private var latestConfidencePct = 0
    private var isLeftHandMode = false

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else resultText.text = "Camera permission denied"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultText = findViewById(R.id.resultText)
        chipGroup = findViewById(R.id.chipGroup)
        clearButton = findViewById(R.id.clearButton)
        doneButton = findViewById(R.id.doneButton)
        navHome = findViewById(R.id.navHome)
        navCamera = findViewById(R.id.navCamera)
        navArchive = findViewById(R.id.navArchive)
        switchLeftHandMode = findViewById(R.id.switchLeftHandMode)

        handLandmarkerHelper = HandLandmarkerHelper(this, this)

        signClassifier = SignClassifier(this) { ready ->
            runOnUiThread {
                resultText.text = if (ready) "Show open palm to start..." else "Model failed to load"
            }
        }

        clearButton.setOnClickListener {
            collectedSigns.clear()
            collectedAccuracies.clear()
            collectedResponseTimes.clear()
            lastAddedLabel = ""
            lastAddedMs = 0L
            isStarted = false
            noHandSinceMs = 0L
            latestConfidencePct = 0
            chipGroup.removeAllViews()
            synchronized(sequenceBuffer) { sequenceBuffer.clear() }
            resultText.text = "Show open palm to start..."
        }

        doneButton.setOnClickListener {
            val sentence = collectedSigns.joinToString(" ")

            val averageAccuracy = if (collectedAccuracies.isNotEmpty()) {
                collectedAccuracies.average().toInt()
            } else {
                0
            }

            val averageResponseTimeMs = if (collectedResponseTimes.isNotEmpty()) {
                collectedResponseTimes.average().toLong()
            } else {
                0L
            }

            val intent = Intent(this, ResultActivity::class.java)
            intent.putExtra("SENTENCE", sentence)
            intent.putExtra("ACCURACY_SCORE", averageAccuracy)
            intent.putExtra("RESPONSE_TIME_MS", averageResponseTimeMs)
            startActivity(intent)
            overridePendingTransition(0, 0)
        }

        navHome.setOnClickListener {
            startActivity(Intent(this, HomeActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }

        navCamera.setOnClickListener {
            // already on capture screen
        }

        navArchive.setOnClickListener {
            // archive screen later
        }

        switchLeftHandMode.setOnCheckedChangeListener { _, isChecked ->
            isLeftHandMode = isChecked
            resultText.text = if (isChecked) {
                "Left-hand mode on"
            } else {
                "Left-hand mode off"
            }
        }

        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun addChip(label: String, index: Int) {
        val chip = Chip(this)
        chip.text = label
        chip.isCloseIconVisible = true
        chip.setOnCloseIconClickListener {
            collectedSigns.removeAt(index)
            refreshChips()
        }
        chipGroup.addView(chip)
    }

    private fun refreshChips() {
        chipGroup.removeAllViews()
        lastAddedLabel = if (collectedSigns.isEmpty()) "" else collectedSigns.last()
        collectedSigns.forEachIndexed { index, label ->
            addChip(label, index)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    if (!signClassifier.ready) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    val now = System.currentTimeMillis()
                    if (now - lastDetectRequestMs < 40L) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    lastDetectRequestMs = now
                    val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
                    val rotated = ImageUtils.rotateBitmap(
                        bitmap, imageProxy.imageInfo.rotationDegrees
                    )
                    val processedFrame = if (isLeftHandMode) {
                        ImageUtils.flipBitmapHorizontally(rotated)
                    } else {
                        rotated
                    }
                    handLandmarkerHelper.detectAsync(processedFrame)
                } finally {
                    imageProxy.close()
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onResults(result: HandLandmarkerResult) {
        val now = System.currentTimeMillis()
        if (now - lastAcceptedMs < 40L) return
        lastAcceptedMs = now

        val hands = result.landmarks()

        // Auto-pause if no hand detected for 3 seconds
        if (isStarted && hands.isEmpty()) {
            if (noHandSinceMs == 0L) {
                noHandSinceMs = now
            } else if (now - noHandSinceMs >= 3000L) {
                isStarted = false
                noHandSinceMs = 0L
                lastAddedLabel = ""
                synchronized(sequenceBuffer) { sequenceBuffer.clear() }
                runOnUiThread {
                    resultText.text = "Show open palm to start..."
                }
            }
            return
        } else {
            noHandSinceMs = 0L
        }

        if (!isStarted) {
            runOnUiThread {
                if (hands.isNotEmpty() && isOpenPalm(hands[0])) {
                    isStarted = true
                    resultText.text = "Go! Show your signs..."
                } else {
                    resultText.text = "Show open palm to start..."
                }
            }
            return
        }

        val feature = SignMath.resultToFeature(result)
        synchronized(sequenceBuffer) {
            if (sequenceBuffer.size == 30) sequenceBuffer.removeFirst()
            sequenceBuffer.addLast(feature)
        }

        if (sequenceBuffer.size < 30 || !signClassifier.ready) return

        val sequence = synchronized(sequenceBuffer) { sequenceBuffer.toTypedArray() }
        val prediction = signClassifier.predict(sequence) ?: return

        runOnUiThread {
            if (prediction.confidence >= 0.75f) {
                val label = prediction.label
                val pct = (prediction.confidence * 100).toInt()
                resultText.text = "$label ($pct%)"
                latestConfidencePct = pct

                if (label != lastAddedLabel) {
                    lastAddedLabel = label
                    lastAddedMs = now
                } else if (now - lastAddedMs >= 1000L &&
                    collectedSigns.lastOrNull() != label) {
                    collectedSigns.add(label)
                    collectedAccuracies.add(pct)
                    collectedResponseTimes.add(now - lastAddedMs)
                    addChip(label, collectedSigns.size - 1)
                }
            } else {
                resultText.text = "Uncertain"
                lastAddedLabel = ""
            }
        }
    }

    private fun isOpenPalm(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size < 21) return false
        val wristY = landmarks[0].y()
        val fingertips = listOf(4, 8, 12, 16, 20)
        val allFingersUp = fingertips.all { landmarks[it].y() < wristY }
        val indexX = landmarks[8].x()
        val pinkyX = landmarks[20].x()
        val spread = Math.abs(indexX - pinkyX) > 0.1f
        return allFingersUp && spread
    }

    override fun onError(message: String) {
        runOnUiThread { resultText.text = "Error: $message" }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handLandmarkerHelper.close()
        signClassifier.close()
    }
}