#!/bin/bash

# Diretório principal
dir="/home/luan/Desktop/PKLot/PKLotSegmented"

# Loop pelas universidades
for universidade in "PUC" "UFPR04" "UFPR05"; do
    dirUniversidade="${dir}/${universidade}"
    
    # Loop pelos climas
    for clima in "Sunny" "Rainy" "Cloudy"; do
        dirClima="${dirUniversidade}/${clima}"
        
        # Loop pelos dias
        for dia in "${dirClima}"/*; do
            # Extrai o nome do dia do diretório
            nomeDia=$(basename "${dia}")
            
            # Diretório do dia na universidade
            dirDia="${dirUniversidade}/${nomeDia}"
            
            # Se diretório não existe, cria
            if [ ! -d "${dirDia}" ]; then
                mkdir "${dirDia}"
                mkdir "${dirDia}/Empty"
                mkdir "${dirDia}/Occupied"
            fi
            
            # Move as fotos da ocupação para o diretório do dia na universidade
            mv "${dia}/Empty"/* "${dirDia}/Empty"
            mv "${dia}/Occupied"/* "${dirDia}/Occupied"
        done
    done

    # Removando climas
    rm -rf "${dirUniversidade}/Sunny"
    rm -rf "${dirUniversidade}/Rainy"
    rm -rf "${dirUniversidade}/Cloudy"
done
