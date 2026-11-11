package com.programacion.taller3.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.util.List;

public class BusquedaSemanticaMain {
    static void main(){

        var model = EmbeddingModelUtil.createModel();

        var documentos = List.of(
          "Java es un lenguaje de programación orientada a objetos",
          "Phyton es un lenguaje popular en inteligencia artificial",
          "LOs embeddings representan texto como vectores"
        );

        List<Embedding> embeddings = documentos.stream().map(model::embed)
                .map(Response::content)
                .toList();

        var consulta = "programación orientada a objetos";
        var embeddingConsulta = model.embed(consulta).content();

        int index = 0;
        int mejorIndice= 0;
        double mejorScore = -1.0;

        for(var it:embeddings) {
            double score = CosineSimilarity.between(embeddingConsulta,it);

            System.out.printf("Documento %d - Similitud: %.4f%n", index, score);

            if (score > mejorScore){
                mejorScore = score;
                mejorIndice = index;
            }
            index ++;
        }
        System.out.printf("Mejor Resultado: [%d] - %s%n", mejorIndice, documentos.get(mejorIndice));
        System.out.printf("Similitud %.4f%n", mejorScore);
    }
}
