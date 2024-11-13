
# Caminho para a pasta que contém os arquivos .yml
$ymlFolder = "configs\new_tests_2"

# Caminho para o script Python
$pythonScript = ".\src\pipeline.py"

# Obter todos os arquivos .yml na pasta
$ymlFiles = Get-ChildItem -Path $ymlFolder -Filter *.yml

# Iterar sobre cada arquivo .yml e executar o comando Python
foreach ($ymlFile in $ymlFiles) {
    $ymlPath = $ymlFile.FullName
    Write-Host "Executando para: $ymlPath"
    python $pythonScript $ymlPath
}
