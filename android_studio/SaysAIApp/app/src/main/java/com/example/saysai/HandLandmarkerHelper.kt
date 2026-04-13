package com.example.saysai

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

class HandLandmarkerHelper(
    private val context: Context,
    private val listener: Listener
) {
    companion object {
        private const val MODEL_PATH = "models/hand_landmarker.task"
    }

    private var handLandmarker: HandLandmarker? = null

    init { setup() }

    private fun setup() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ -> listener.onResults(result) }
            .setErrorListener { error -> listener.onError(error.message ?: "MediaPipe error") }
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun detectAsync(bitmap: Bitmap) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker?.detectAsync(mpImage, SystemClock.uptimeMillis())
    }

    fun close() {
        handLandmarker?.close()
        handLandmarker = null
    }

    interface Listener {
        fun onResults(result: HandLandmarkerResult)
        fun onError(message: String)
    }
}