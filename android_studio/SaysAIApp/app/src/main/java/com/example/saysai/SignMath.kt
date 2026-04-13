package com.example.saysai

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.sqrt

object SignMath {

    fun resultToFeature(result: HandLandmarkerResult): FloatArray {
        val left = FloatArray(63)
        val right = FloatArray(63)

        val hands = result.landmarks()
        val handedness = result.handedness()

        for (i in hands.indices) {
            val coords = normalizeHand(hands[i])
            val label = handedness.getOrNull(i)
                ?.firstOrNull()
                ?.categoryName()
                ?.uppercase() ?: ""

            when (label) {
                "LEFT" -> System.arraycopy(coords, 0, left, 0, 63)
                "RIGHT" -> System.arraycopy(coords, 0, right, 0, 63)
                else -> {
                    if (left.all { it == 0f }) {
                        System.arraycopy(coords, 0, left, 0, 63)
                    } else {
                        System.arraycopy(coords, 0, right, 0, 63)
                    }
                }
            }
        }

        val merged = FloatArray(126)
        System.arraycopy(left, 0, merged, 0, 63)
        System.arraycopy(right, 0, merged, 63, 63)
        return merged
    }

    private fun normalizeHand(landmarks: List<NormalizedLandmark>): FloatArray {
        val arr = Array(21) { FloatArray(3) }

        for (i in landmarks.indices) {
            arr[i][0] = landmarks[i].x()
            arr[i][1] = landmarks[i].y()
            arr[i][2] = landmarks[i].z()
        }

        val wristX = arr[0][0]
        val wristY = arr[0][1]
        val wristZ = arr[0][2]

        for (i in 0 until 21) {
            arr[i][0] -= wristX
            arr[i][1] -= wristY
            arr[i][2] -= wristZ
        }

        var scale = norm(arr[9][0], arr[9][1], arr[9][2])
        if (scale < 1e-6f) scale = norm(arr[12][0], arr[12][1], arr[12][2])
        if (scale < 1e-6f) scale = 1f

        val flat = FloatArray(63)
        var idx = 0
        for (i in 0 until 21) {
            flat[idx++] = arr[i][0] / scale
            flat[idx++] = arr[i][1] / scale
            flat[idx++] = arr[i][2] / scale
        }
        return flat
    }

    private fun norm(x: Float, y: Float, z: Float): Float {
        return sqrt(x * x + y * y + z * z)
    }
}