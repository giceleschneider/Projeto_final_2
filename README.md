# Board Trello

https://trello.com/invite/b/6aea673a7ff8140bce3950fb/ATTIbffd916d9c5d405de58e82e3e3313baaF7DF0EDA/ppe-detector-api

# PPE Detector API

Projeto acadêmico de **detecção de objetos** usando **YOLOv8 + FastAPI**.

A ideia é detectar equipamentos e elementos de segurança em imagens de ambientes de construção civil e disponibilizar o modelo por meio de uma API REST.

## 1. O problema

Em um canteiro de obras, é importante identificar pessoas e equipamentos de proteção individual (EPIs). O modelo deste projeto tenta localizar objetos como capacete, máscara e colete de segurança.

A API recebe uma imagem e retorna somente os dados das detecções em JSON.

---

## 2. Dataset escolhido

Foi escolhido o **Construction Site Safety Image Dataset**, disponibilizado no Kaggle a partir de um conjunto em formato YOLOv8.

O dataset é interessante para este projeto porque já possui anotações no formato YOLO e possui exatamente **10 classes**, atendendo ao requisito mínimo do desafio.

### Classes

1. Hardhat
2. Mask
3. NO-Hardhat
4. NO-Mask
5. NO-Safety Vest
6. Person
7. Safety Cone
8. Safety Vest
9. machinery
10. vehicle

Uma referência do dataset no Kaggle pode ser encontrada em:

https://www.kaggle.com/code/daffamuhamadazhar/yolov8-ppe-detection/input

A página informa que o dataset possui 10 classes e anotações em formato YOLOv8.

---

## 3. Modelo utilizado

Foi utilizado o **YOLOv8n (YOLOv8 Nano)**.

Escolhi o YOLOv8n porque é uma versão menor da família YOLO e é mais adequada para um projeto de aprendizagem, pois exige menos recursos computacionais do que modelos maiores.

O treinamento utiliza um modelo pré-treinado:

```python
model = YOLO("yolov8n.pt")
```

Depois o modelo é ajustado para as 10 classes do dataset.

A documentação da Ultralytics apresenta o treinamento de modelos YOLO a partir de pesos pré-treinados como uma das formas recomendadas de começar o treinamento.

---

## 4. Estrutura do projeto

```text
ppe_detector_api/
│
├── main.py
├── train.py
├── validate.py
├── data.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── best.pt
│
├── data/
│   └── construction-site-safety/
│       ├── train/
│       ├── valid/
│       └── test/
│
└── runs/
```

A pasta `data/` e os arquivos `.pt` não devem ser enviados para o GitHub neste projeto, por isso estão no `.gitignore`.

---

# 5. Instalação

## 5.1 Criar o ambiente virtual

No terminal do VS Code:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 5.2 Instalar as bibliotecas

```bash
pip install -r requirements.txt
```

---

# 6. Preparação do dataset

Baixe o dataset escolhido no Kaggle e extraia os arquivos dentro de:

```text
data/construction-site-safety/
```

A estrutura esperada é:

```text
data/
└── construction-site-safety/
    ├── train/
    │   ├── images/
    │   └── labels/
    │
    ├── valid/
    │   ├── images/
    │   └── labels/
    │
    └── test/
        ├── images/
        └── labels/
```

Se o nome ou a estrutura das pastas for diferente, altere o arquivo `data.yaml`.

---

# 7. Treinamento

Com o ambiente virtual ativado:

```bash
python train.py
```

O treinamento utiliza:

- YOLOv8n
- 30 épocas
- imagens de 640 pixels
- batch de 8
- early stopping com `patience=10`

Esses valores foram escolhidos para manter o projeto simples.

Se o computador tiver pouca memória, diminua o `batch` para 4 ou 2 no arquivo `train.py`.

Depois do treinamento, o melhor modelo será copiado automaticamente para:

```text
models/best.pt
```

---

# 8. Validação

Depois do treinamento:

```bash
python validate.py
```

O programa mostrará métricas como:

```text
mAP50: 0.xxxx
mAP50-95: 0.xxxx
```

Essas métricas ajudam a avaliar o desempenho do detector.

---

# 9. Executando a API

Com o modelo `models/best.pt` disponível:

```bash
uvicorn main:app --reload
```

A API ficará disponível normalmente em:

```text
http://127.0.0.1:8000
```

A documentação automática do FastAPI pode ser acessada em:

```text
http://127.0.0.1:8000/docs
```

---

# 10. Endpoint de detecção

O endpoint principal é:

```text
POST /detect
```

Ele recebe uma imagem através de `UploadFile`.

No Swagger:

1. Abra `/docs`.
2. Localize `POST /detect`.
3. Clique em **Try it out**.
4. Escolha uma imagem.
5. Clique em **Execute**.

A API não devolve uma imagem desenhada. Ela retorna apenas os dados das detecções.

---

# 11. Exemplo de resposta

```json
{
  "status": "success",
  "detections": [
    {
      "class_name": "Hardhat",
      "confidence": 0.92,
      "box": {
        "x_min": 150,
        "y_min": 200,
        "x_max": 250,
        "y_max": 300
      }
    },
    {
      "class_name": "Person",
      "confidence": 0.88,
      "box": {
        "x_min": 40,
        "y_min": 50,
        "x_max": 300,
        "y_max": 420
      }
    }
  ]
}
```

### Significado dos campos

`class_name` = classe detectada.

`confidence` = confiança do modelo naquela detecção.

`box` = coordenadas da caixa delimitadora.

- `x_min`: esquerda
- `y_min`: topo
- `x_max`: direita
- `y_max`: baixo

---

# 12. Testando pelo Postman

Configure:

```text
POST http://127.0.0.1:8000/detect
```

Em **Body**:

```text
form-data
```

Crie uma chave:

```text
file
```

Altere o tipo dessa chave para:

```text
File
```

Escolha uma imagem e envie.

---

# 13. Arquivos principais

### `train.py`

Responsável pelo treinamento do YOLOv8.

### `validate.py`

Responsável pela validação do modelo.

### `main.py`

Responsável pela API FastAPI e pelo endpoint `/detect`.

### `data.yaml`

Informa ao YOLO onde estão as imagens e quais são as 10 classes.

### `requirements.txt`

Lista as bibliotecas necessárias para executar o projeto.

---

# 14. Metodologia ágil - Kanban

Para atender ao requisito de metodologia ágil, foi utilizado um quadro Kanban no Trello.

Sugestão de colunas:

```text
BACKLOG
   ↓
A FAZER
   ↓
EM ANDAMENTO
   ↓
EM TESTE
   ↓
CONCLUÍDO
```

### Cards sugeridos

**Backlog**

- Escolher dataset
- Criar repositório GitHub
- Criar ambiente Python

**A Fazer**

- Baixar dataset
- Configurar data.yaml
- Instalar dependências
- Criar código de treinamento
- Criar API

**Em Andamento**

- Treinar YOLOv8
- Avaliar resultados

**Em Teste**

- Testar endpoint `/detect`
- Testar imagens diferentes
- Testar Swagger
- Testar Postman

**Concluído**

- Dataset preparado
- Modelo treinado
- API funcionando
- README criado
- Projeto publicado no GitHub

---

# 15. Tecnologias utilizadas

- Python
- YOLOv8
- Ultralytics
- FastAPI
- Uvicorn
- Pillow
- Kaggle
- Postman
- Swagger/OpenAPI
- Git e GitHub
- Trello

---

# 16. Fluxo da aplicação

```text
                 DATASET
                    |
                    v
             Preparação dos dados
                    |
                    v
              YOLOv8 pré-treinado
                    |
                    v
                Treinamento
                    |
                    v
               best.pt
                    |
                    v
              FastAPI /detect
                    |
                    v
             Upload da imagem
                    |
                    v
               YOLOv8
                    |
                    v
               Detecções
                    |
                    v
              JSON estruturado
```

---

# 17. Observação sobre o projeto

Este projeto foi desenvolvido com foco em aprendizado e simplicidade. A intenção não é criar um sistema de produção, mas demonstrar o fluxo completo solicitado no desafio:

**dataset → treinamento → validação → modelo → API → JSON**

Para uma aplicação real, poderiam ser adicionados autenticação, tratamento mais completo de erros, Docker, testes automatizados, logging, monitoramento e deploy em nuvem.

---

## Referências

- Kaggle - Construction Site Safety / PPE Detection:
  https://www.kaggle.com/code/daffamuhamadazhar/yolov8-ppe-detection/input

- Ultralytics - documentação de treinamento:
  https://docs.ultralytics.com/modes/train

- FastAPI:
  https://fastapi.tiangolo.com/
