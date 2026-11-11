package com.programacion.taller3.embeddings;

import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;

import java.nio.file.Path;
import java.nio.file.Paths;

public final class EmbeddingModelUtil {

    private static final Path PATH_MODEL = Paths.get(
            "/home/labcom/Documentos/taller 3/02.embeddings/modelos/multilingual-e5-small.onnx");
    private static final Path PATH_TOKENIZER = Paths.get(
            "/home/labcom/Documentos/taller 3/02.embeddings/modelos/multilingual-e5-small.json");

    private EmbeddingModelUtil() {
    }

    public static OnnxEmbeddingModel createModel() {
        return new OnnxEmbeddingModel(PATH_MODEL, PATH_TOKENIZER, PoolingMode.MEAN);
    }
}
