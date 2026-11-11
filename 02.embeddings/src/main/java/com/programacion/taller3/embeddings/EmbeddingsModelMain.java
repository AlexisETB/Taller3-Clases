package com.programacion.taller3.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;

public class EmbeddingsModelMain {
    static void main() {
        var model = EmbeddingModelUtil.createModel();

        var text = "Hola, cómo estás?";

        Response<Embedding> respose = model.embed(text);
        Embedding embedding = respose.content();

        System.out.println("Dimension of the embedding: " + embedding.dimension());
        System.out.println("embedding vector: " + embedding.vectorAsList());
    }
}
