package com.sq.rknn.rknnyolov8;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ImgAnalyzer implements ImageAnalysis.Analyzer {

    public static class Result {
        public Bitmap outputBitmap;
        public long processTimeMs;
        public List<DetectionResult.ObjectResult> objectResults;

        public Result(long l, Bitmap outputImageBitmap, List<DetectionResult.ObjectResult> objectResults) {
            this.processTimeMs = l;
            this.outputBitmap = outputImageBitmap;
            this.objectResults = objectResults;
        }
    }
    private final YoloV8Detect yolov8Detect;
    private final Callback callBack;

    Set<String> resultSet = new HashSet<>();

    public ImgAnalyzer(YoloV8Detect yolov8Detect, Callback callBack) {
        this.yolov8Detect = yolov8Detect;
        this.callBack = callBack;

        resultSet.add("image_out");
        resultSet.add("scaled_box_out_next");
    }

    public Bitmap rotateBitmap(Bitmap bitmap, float degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    @Override
    public void analyze(ImageProxy image) {

        AsyncTask.execute(() -> {
            long startTime = SystemClock.uptimeMillis();
            Bitmap imgBitmap = image.toBitmap();
            Bitmap rawBitmap = compressBitmapTo640x640(imgBitmap);
            Bitmap bitmap = rotateBitmap(rawBitmap, 90);
            if (bitmap != null) {
                DetectionResult result = (DetectionResult) yolov8Detect.detect2(bitmap);
                if (result != null) {
                    for (int i = 0; i < result.objectResults.size(); i++) {
                        Log.e("jack", "###@@@ result = " + result.objectResults.get(i).classId + ", "
                                + result.objectResults.get(i).score + ", size = " + result.objectResults.size());
                    }
                    callBack.invoke(new Result(SystemClock.uptimeMillis() - startTime, bitmap, result.objectResults));
                }
            }

            image.close();
        });
    }

    public Bitmap compressBitmapTo640x640(Bitmap originalBitmap) {
        int originalWidth = originalBitmap.getWidth();
        int originalHeight = originalBitmap.getHeight();

        // 计算宽高比
        float ratio = (float) originalWidth / originalHeight;

        int newWidth;
        int newHeight;

        // 根据宽高比计算缩放后的宽高
        if (ratio > 1) {
            newWidth = 640;
            newHeight = (int) (640 / ratio);
        } else {
            newHeight = 640;
            newWidth = (int) (640 * ratio);
        }

        // 创建缩放矩阵
        Matrix matrix = new Matrix();
        matrix.postScale((float) newWidth / originalWidth, (float) newHeight / originalHeight);

        // 进行缩放
        return Bitmap.createBitmap(originalBitmap, 0, 0, originalWidth, originalHeight, matrix, true);
    }

    public byte[] convertBitmapToByteArray(Bitmap bitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream); // 可以选择其他格式，如JPEG
        return outputStream.toByteArray();
    }

    private Bitmap byteArrayToBitmap(byte[] data) {
        return BitmapFactory.decodeByteArray(data, 0, data.length);
    }

    @Override
    protected void finalize() {
    }

    public interface Callback {
        void invoke(Result result);
    }
}
