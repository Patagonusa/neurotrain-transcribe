package com.neurotrain.transcribe

import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class TranscribeAPI {
    companion object {
        // This will be your Render URL
        private const val BASE_URL = "https://neurotrain-transcribe.onrender.com"
        
        private val client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
        
        fun transcribeAudio(audioFile: File, callback: (Result<TranscriptionResult>) -> Unit) {
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    audioFile.name,
                    audioFile.asRequestBody("audio/ogg".toMediaType())
                )
                .build()
            
            val request = Request.Builder()
                .url("$BASE_URL/transcribe")
                .post(requestBody)
                .build()
            
            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: java.io.IOException) {
                    callback(Result.failure(e))
                }
                
                override fun onResponse(call: Call, response: Response) {
                    try {
                        val json = JSONObject(response.body?.string() ?: "{}")
                        val result = TranscriptionResult(
                            transcript = json.getString("transcript"),
                            language = json.getString("language"),
                            tldr = json.getString("tldr")
                        )
                        callback(Result.success(result))
                    } catch (e: Exception) {
                        callback(Result.failure(e))
                    }
                }
            })
        }
    }
}

data class TranscriptionResult(
    val transcript: String,
    val language: String,
    val tldr: String
)
