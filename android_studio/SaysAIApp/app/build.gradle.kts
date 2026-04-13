plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.saysai"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.saysai"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    dependencies {
        implementation("androidx.core:core-ktx:1.13.1")
        implementation("androidx.appcompat:appcompat:1.7.0")
        implementation("com.google.android.material:material:1.12.0")
        implementation("androidx.constraintlayout:constraintlayout:2.1.4")

        implementation("androidx.camera:camera-core:1.3.4")
        implementation("androidx.camera:camera-camera2:1.3.4")
        implementation("androidx.camera:camera-lifecycle:1.3.4")
        implementation("androidx.camera:camera-view:1.3.4")

        implementation("com.google.mediapipe:tasks-vision:0.10.14")

        implementation("com.google.android.gms:play-services-tflite-java:16.4.0")
        implementation("com.google.android.gms:play-services-tflite-support:16.4.0")

        testImplementation("junit:junit:4.13.2")
        androidTestImplementation("androidx.test.ext:junit:1.2.1")
        androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    }
}