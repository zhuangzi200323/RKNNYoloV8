package com.sq.rknn.rknnyolov8;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.sq.rknn.rknnyolov8.databinding.ActivityMainBinding;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};
    private ActivityMainBinding binding;
    private ExecutorService backgroundExecutor = Executors.newCachedThreadPool();// newSingleThreadExecutor();
    private List<String> labelData;
    private ImageCapture imageCapture;
    private ImageAnalysis imageAnalysis;

    private YoloV8Detect yolov8Detect = new YoloV8Detect();

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        paint.setColor(Color.BLACK);
        paint.setTextSize(20f);
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_OVER));
        try {
            String labelFile = "coco_80_labels_list.txt";
            labelData = readLabels(labelFile);
            boolean retInit = yolov8Detect.init(AssetHelper.assetFilePath(this, "yolov8n.rknn"),
                    AssetHelper.assetFilePath(this, labelFile), false);
            if (!retInit) {
                Log.e(TAG, "YoloV8Detect Init failed");
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets", e);
        }

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {
        ListenableFuture cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = (ProcessCameraProvider) cameraProviderFuture.get();

                Preview preview = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        //.setTargetRotation(Surface.ROTATION_180) //120cm的机器3399上，需要旋转
                        .build();

                preview.setSurfaceProvider(binding.viewFinder.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .build();

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                cameraProvider.unbindAll();

                cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageCapture, imageAnalysis);

                setImgAnalyzer();
            } catch (Exception exc) {
                Log.e(TAG, "Use case binding failed", exc);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void updateUI(ImgAnalyzer.Result result) {
        if (result == null) {
            return;
        }
        runOnUiThread(() -> {
            binding.inferenceTimeValue.setText(result.processTimeMs + "ms");
            showBitmap(result);
            StringBuilder textBuf = new StringBuilder();
            for (DetectionResult.ObjectResult boxInfo : result.objectResults) {
                textBuf.append(String.format("%s", labelData.get(boxInfo.classId)));
                textBuf.append("->");
                textBuf.append(String.format("%.2f", boxInfo.score));
                textBuf.append("; ");
            }
            binding.detectedItem1.setText(textBuf);
        });
    }
    Paint paint = new Paint();

    private void showBitmap(ImgAnalyzer.Result result) {
        if (result.outputBitmap == null) return;

//        Bitmap mutableBitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true);
//        Canvas canvas = new Canvas(mutableBitmap);
//
//        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint);
//        for (DetectionResult.ObjectResult boxInfo : result.objectResults) {
//            canvas.drawText(String.format("%s:%.2f", labelData.get(boxInfo.classId), boxInfo.score),
//                    boxInfo.x1 - boxInfo.x2 / 2, boxInfo.y1 - boxInfo.y2 / 2, paint);
//        }
        binding.resultIv.setImageBitmap(result.outputBitmap);
    }

    private List<String> readLabels(String labelFile) {
        List<String> labels = new ArrayList<>();
        try {
            InputStream inputStream = getAssets().open(labelFile);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                labels.add(line);
            }
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;
    }

//    private byte[] readModel() {
//        try {
//            int modelID = R.raw.yolov5n_with_pre_post_processing;
//            InputStream inputStream = getResources().openRawResource(modelID);
//            byte[] buffer = new byte[inputStream.available()];
//            inputStream.read(buffer);
//            inputStream.close();
//            return buffer;
//        } catch (IOException e) {
//            e.printStackTrace();
//            return null;
//        }
//    }

    private void setImgAnalyzer() {
        imageAnalysis.clearAnalyzer();
        imageAnalysis.setAnalyzer(backgroundExecutor, new ImgAnalyzer(yolov8Detect, this::updateUI));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
        boolean retInit = yolov8Detect.release();
        if (!retInit) {
            Log.e(TAG, "YoloV8Detect Release failed");
        }
    }
}