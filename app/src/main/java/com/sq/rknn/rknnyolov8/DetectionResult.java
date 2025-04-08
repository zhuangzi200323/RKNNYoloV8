package com.sq.rknn.rknnyolov8;

import java.util.ArrayList;
import java.util.List;

public class DetectionResult {
    public static class ObjectResult {
        public float x1;
        public float y1;
        public float x2;
        public float y2;
        public int classId;
        public float score;

        public ObjectResult(float x1, float y1, float x2, float y2, int classId, float score) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.classId = classId;
            this.score = score;
        }
    }

    public List<ObjectResult> objectResults;

    public DetectionResult() {
        this.objectResults = new ArrayList<>();
    }

    public void addObjectResult(ObjectResult result) {
        objectResults.add(result);
    }
}