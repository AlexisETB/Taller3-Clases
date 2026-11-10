package com.programacion.taller3.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.output.Response;

import java.nio.file.Paths;

public class EmbeddingsModelMain {
    static void main() {
        var pathModel = Paths.get("/home/labcom/Documentos/taller 3/02.embeddings/modelos/multilingual-e5-small.onnx");
        var pathToTokenizer = Paths.get("/home/labcom/Documentos/taller 3/02.embeddings/modelos/multilingual-e5-small.json");

        OnnxEmbeddingModel model = new OnnxEmbeddingModel(pathModel, pathToTokenizer, PoolingMode.MEAN);

        var text = "Hola, cómo estás?";

        Response<Embedding> respose = model.embed(text);
        Embedding embedding = respose.content();

        System.out.println("Dimension of the embedding: " + embedding.dimension());
        System.out.println("embedding vector: " + embedding.vectorAsList());
    }
}
