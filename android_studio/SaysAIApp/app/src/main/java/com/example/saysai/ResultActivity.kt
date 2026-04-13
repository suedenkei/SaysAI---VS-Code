package com.example.saysai

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val sentence = intent.getStringExtra("SENTENCE") ?: "(nothing signed)"
        val accuracyScore = intent.getIntExtra("ACCURACY_SCORE", 0)
        val responseTimeMs = intent.getLongExtra("RESPONSE_TIME_MS", 0L)

        val resultSentence = findViewById<TextView>(R.id.resultSentence)
        val confidenceText = findViewById<TextView>(R.id.confidenceText)
        val responseTimeText = findViewById<TextView>(R.id.responseTimeText)
        val btnBack = findViewById<ImageView>(R.id.btn_back)
        val btnTranslate = findViewById<ImageView>(R.id.btn_translate)
        val btnCopy = findViewById<ImageView>(R.id.btn_copy)
        val recordMoreButton = findViewById<MaterialButton>(R.id.recordMoreButton)
        val finishButton = findViewById<MaterialButton>(R.id.finishButton)
        val navHome = findViewById<ImageView>(R.id.navHome)
        val navCamera = findViewById<ImageView>(R.id.navCamera)
        val navArchive = findViewById<ImageView>(R.id.navArchive)

        resultSentence.text = sentence
        confidenceText.text = "Accuracy Score: ${accuracyScore}%"
        responseTimeText.text = "Response Time: ${responseTimeMs} ms"

        btnBack.setOnClickListener {
            overridePendingTransition(0, 0)
            finish()
        }

        btnTranslate.setOnClickListener {
            Toast.makeText(this, "Language translation will be added next.", Toast.LENGTH_SHORT).show()
        }

        btnCopy.setOnClickListener {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("Translated Text", sentence)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(this, "Copied to clipboard", Toast.LENGTH_SHORT).show()
        }

        recordMoreButton.setOnClickListener {
            overridePendingTransition(0, 0)
            finish()
        }

        finishButton.setOnClickListener {
            showSaveDialog()
        }

        navHome.setOnClickListener {
            startActivity(Intent(this, HomeActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }

        navCamera.setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }

        navArchive.setOnClickListener {
            Toast.makeText(this, "Archive screen will be added next.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showSaveDialog() {
        AlertDialog.Builder(this)
            .setTitle("Save translation?")
            .setMessage("Do you want to store this translated text and video in your archive?")
            .setCancelable(true)
            .setNegativeButton("Discard") { dialog, _ ->
                dialog.dismiss()
                finish()
            }
            .setPositiveButton("Save") { dialog, _ ->
                Toast.makeText(this, "Saved to archive (placeholder)", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
                finish()
            }
            .show()
    }
}