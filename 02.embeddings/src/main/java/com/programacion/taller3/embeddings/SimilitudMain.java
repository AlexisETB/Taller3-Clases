package com.programacion.taller3.embeddings;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.CosineSimilarity;

public class SimilitudMain {
    static void main() {
        var model = EmbeddingModelUtil.createModel();

        var emmbeding1 = model.embed("Cuál es el area de un circulo?").content();
        var emmbeding2 = model.embed("Cuál es el area de un cuadrado?").content();
        var emmbeding3 = model.embed("Qué hora es?").content();
        var emmbeding4 = model.embed("El area es redonda").content();

        //CAlcular la similitud entre los embeddings
        double sim12 = CosineSimilarity.between(emmbeding1, emmbeding2);
        double sim13 = CosineSimilarity.between(emmbeding1, emmbeding3);
        double sim23 = CosineSimilarity.between(emmbeding2, emmbeding3);
        double sim14 = CosineSimilarity.between(emmbeding1, emmbeding4);

        System.out.println("SImilitud emb1 y emb2: " + sim12);
        System.out.println("SImilitud emb1 y emb3: " + sim13);
        System.out.println("SImilitud emb2 y emb3: " + sim23);
        System.out.println("SImilitud emb1 y emb4: " + sim14);


    }
}
