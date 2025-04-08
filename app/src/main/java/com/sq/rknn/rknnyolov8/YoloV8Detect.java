package com.sq.rknn.rknnyolov8;

import android.graphics.Bitmap;

public class YoloV8Detect {
    static {
        System.loadLibrary("rknn_yolov8");
    }

    public native boolean init(String modelPath, String labelListPath, boolean useZeroCopy);

    public native boolean detect(Bitmap srtBitmap);
    public native Object detect2(Bitmap srtBitmap);

    public native boolean release();
}
