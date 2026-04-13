package com.example.saysai

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        val navHome = findViewById<ImageView>(R.id.navHome)
        val navCamera = findViewById<ImageView>(R.id.navCamera)
        val navArchive = findViewById<ImageView>(R.id.navArchive)
        val cardFsl = findViewById<View>(R.id.cardFsl)
        val cardBsl = findViewById<View>(R.id.cardBsl)
        val cardAsl = findViewById<View>(R.id.cardAsl)
        val tvSeeAll = findViewById<TextView>(R.id.tvSeeAll)
        val tutorialOverlay = findViewById<View>(R.id.tutorialOverlay)
        val tutorialBackButton = findViewById<ImageView>(R.id.tutorialBackButton)
        val overlayCardFsl = findViewById<View>(R.id.overlayCardFsl)
        val overlayCardBsl = findViewById<View>(R.id.overlayCardBsl)
        val overlayCardAsl = findViewById<View>(R.id.overlayCardAsl)
        val splashOverlay = findViewById<FrameLayout>(R.id.splashOverlay)
        val splashLogo = findViewById<ImageView>(R.id.splashLogo)
        val launchedFromLauncher = intent?.action == Intent.ACTION_MAIN && intent.hasCategory(Intent.CATEGORY_LAUNCHER)

        if (launchedFromLauncher) {
            splashLogo.animate()
                .alpha(1f)
                .setDuration(500)
                .withEndAction {
                    splashLogo.animate()
                        .alpha(0f)
                        .setDuration(500)
                        .setStartDelay(350)
                        .withEndAction {
                            splashOverlay.visibility = View.GONE
                        }
                        .start()
                }
                .start()
        } else {
            splashOverlay.visibility = View.GONE
        }

        navHome.setOnClickListener {
            // already on home
        }

        navCamera.setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
            overridePendingTransition(0, 0)
        }

        navArchive.setOnClickListener {
            // archive screen later
        }

        cardFsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLkEbhtuT-Wbr9IZ7RJqlwsBqZvAvh__bO&si=7j3fCqMnUcszCSmG")
            )
            startActivity(intent)
        }

        cardBsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLAoW1zMlmOlldpjDNkbumwGkhz5zOh1zQ&si=6OJqYfqPPfkluR9f")
            )
            startActivity(intent)
        }

        cardAsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLMN7QCuj6dfaUwmtdkdKhINGZzyGwp7Q1&si=C2QvElPB93nE35xq")
            )
            startActivity(intent)
        }

        overlayCardFsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLkEbhtuT-Wbr9IZ7RJqlwsBqZvAvh__bO&si=7j3fCqMnUcszCSmG")
            )
            startActivity(intent)
        }

        overlayCardBsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLAoW1zMlmOlldpjDNkbumwGkhz5zOh1zQ&si=6OJqYfqPPfkluR9f")
            )
            startActivity(intent)
        }

        overlayCardAsl.setOnClickListener {
            val intent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://youtube.com/playlist?list=PLMN7QCuj6dfaUwmtdkdKhINGZzyGwp7Q1&si=C2QvElPB93nE35xq")
            )
            startActivity(intent)
        }

        tvSeeAll.setOnClickListener {
            tutorialOverlay.visibility = View.VISIBLE
        }

        tutorialBackButton.setOnClickListener {
            tutorialOverlay.visibility = View.GONE
        }
    }
}