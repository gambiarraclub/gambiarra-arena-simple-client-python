# Gambiarra Arena - Cliente Python

## Como funciona o jogo?

A Gambiarra Arena é uma **competição ao vivo entre IAs locais**. Cada participante traz seu computador com um modelo de IA rodando localmente e compete para dar a melhor resposta a desafios.

```
┌─────────────────────────────────────────────────────────────────┐
│                      COMO FUNCIONA                              │
│                                                                 │
│  1. Um desafio é criado (ex: "Escreva uma poesia                │
│     sobre o sertão") e envia para todos ao mesmo tempo          │
│                                                                 │
│  2. O seu cliente recebe o desafio automaticamente e            │
│     repassa para o modelo de IA rodando no seu computador       │
│                                                                 │
│  3. O modelo gera a resposta, e o cliente envia palavra por     │
│     palavra para o telão em tempo real                          │
│                                                                 │
│  4. Os participantes votam na melhor resposta                   │
└─────────────────────────────────────────────────────────────────┘
```

**O que você controla (sua "gambiarra"):**
- Qual **modelo de IA** usar (menor e mais rápido? maior e mais inteligente?)
- Como **instruir** o modelo antes de ele receber o desafio (o "system prompt")
- Os **parâmetros** do modelo (mais criativo? mais preciso? mais rápido?)

A graça do jogo é configurar seu setup de forma diferente para tentar dar a melhor resposta.

## Arquivos do projeto

| Arquivo | O que faz |
|---------|-----------|
| `gambiarra-arena-client.py` | Cliente simples — chama o Ollama diretamente. |
| `gambiarra-arena-client-langchain.py` | Cliente avançado — usa LangChain para montar prompts mais elaborados. |
| `orchestration.py` | Módulo usado pelo cliente LangChain com templates e estratégias. |

## Instalação

```bash
python3 -m venv .venv #criar ambiente virtual python
source .venv/bin/activate #ativiar o ambiente virtual
pip install -r requirements.txt
```

## Como jogar

### Passo 1: Tenha o Ollama rodando com um modelo

O [Ollama](https://ollama.ai) é o programa que roda a IA localmente no seu computador. Instale-o e baixe um modelo:

```bash
# Baixar um modelo (escolha um que caiba na sua máquina)
ollama pull qwen3:0.6b      # Pequeno e rápido (~400MB)
ollama pull gemma3:1b        # Médio (~800MB)
ollama pull llama3.2         # Maior e mais capaz (~2GB)
```

### Passo 2: Configure o cliente

Abra o arquivo do cliente que quiser usar e edite as variáveis no topo:

```python
# === CONEXÃO (o organizador vai informar esses dados) ===
HOST = "192.168.0.212"      # IP do servidor da arena (rede local)
PIN = "937414"              # PIN da sessão (muda a cada encontro)

# === SUA IDENTIDADE ===
PARTICIPANT_ID = "meuId"    # Um ID único para você (sem espaços)
NICKNAME = "Meu Apelido"   # Seu nome no telão

# === SEU MODELO ===
OLLAMA_MODEL = "qwen3:0.6b" # O modelo que você baixou no Passo 1
```

### Passo 3: Execute o cliente

```bash
# Cliente simples (recomendado para iniciantes)
python gambiarra-arena-client.py

# OU cliente com LangChain (mais opções de customização)
python gambiarra-arena-client-langchain.py
```

Pronto! O cliente vai se conectar, e quando o organizador iniciar uma rodada, o desafio chega automaticamente.

## Qual cliente usar?

### Cliente simples (`gambiarra-arena-client.py`)

Chama o Ollama diretamente. Tem menos dependências e é mais fácil de entender e modificar.

Você customiza o comportamento do modelo editando a função `context_engineering()`:

```python
def context_engineering(prompt: str) -> str:
    system_instructions = """Você é um assistente prestativo.
Responda sempre em português.
Seja direto e criativo.
"""
    return system_instructions + prompt
```

Essa função adiciona suas instruções **antes** do desafio que veio do servidor.

### Cliente LangChain (`gambiarra-arena-client-langchain.py`)

Usa a biblioteca LangChain para montar prompts mais elaborados. Oferece:

- **Templates profissionais** de prompt
- **Estratégias prontas** para diferentes tipos de desafio
- Mais opções de configuração

Configurações extras no topo do arquivo:

```python
USE_ENHANCED = True           # Usar estratégias prontas
TEMPERATURE = 0.7             # Criatividade (0.0 = preciso, 1.0 = criativo)
MAX_TOKENS = 500              # Limite de palavras na resposta
STRATEGY = "accuracy_focused" # Estratégia (veja abaixo)
```

## Estratégias e quando usar cada uma

Se estiver usando o cliente LangChain com `USE_ENHANCED = True`, você pode escolher uma estratégia:

| Estratégia | Temperatura | Limite de tokens | Melhor para |
|-----------|-------------|------------------|-------------|
| `speed_focused` | 0.3 (mais previsível) | 200 (respostas curtas) | Quando velocidade importa mais |
| `accuracy_focused` | 0.5 (equilibrado) | 500 (respostas médias) | Perguntas factuais, quizzes |
| `detailed` | 0.8 (mais criativo) | 1000 (respostas longas) | Poesia, histórias, explicações |

## Glossário rápido

Se você não tem familiaridade com esses termos:

| Termo | O que é |
|-------|---------|
| **Modelo / LLM** | O programa de inteligência artificial que gera texto. Roda no seu computador via Ollama. |
| **Prompt** | O texto que você envia para a IA. Na arena, é o desafio + suas instruções. |
| **System prompt** | Instruções que você dá para a IA *antes* do desafio, definindo como ela deve se comportar. |
| **Temperatura** | Controla a criatividade. Baixa (0.0) = respostas mais "seguras". Alta (1.0) = mais arriscadas e criativas. |
| **Tokens** | Pedacinhos de palavras que a IA gera. Pense como sílabas. "Inteligência" pode ser 3-4 tokens. |
| **Streaming** | A IA envia a resposta palavra por palavra conforme vai gerando, em vez de esperar terminar tudo. |
| **Ollama** | Programa que roda modelos de IA localmente no seu computador, sem precisar de internet. |
| **LangChain** | Biblioteca Python que ajuda a montar prompts mais elaborados e conectar ferramentas à IA. |

## Personalização avançada

### Modificar o template do prompt (cliente LangChain)

Edite os templates em `orchestration.py`:

```python
self.prompt_template = PromptTemplate(
    input_variables=["user_input", "context"],
    template="""Você é um poeta nordestino irreverente.
Responda sempre com humor e referências culturais do sertão.

Contexto: {context}
Pergunta: {user_input}

Resposta:"""
)
```

### Ajustar parâmetros do modelo

Além de temperatura e max_tokens, você pode ajustar outros parâmetros
diretamente no código do `orchestration.py`, na criação do objeto `Ollama`:

```python
llm = Ollama(
    model=self.ollama_model,
    temperature=0.7,        # Criatividade
    num_predict=500,        # Limite de tokens
    top_k=40,               # Considera só os 40 tokens mais prováveis
    top_p=0.9,              # Nucleus sampling (alternativa ao top_k)
    repeat_penalty=1.1,     # Penaliza repetição de palavras
    num_ctx=4096,           # Tamanho da "memória" de contexto
)
```

### Criar novas estratégias (cliente LangChain)

No método `create_enhanced_chain` em `orchestration.py`:

```python
if strategy == "poeta_nordestino":
    temp = 0.9
    max_tok = 800
```

E use no cliente:

```python
STRATEGY = "poeta_nordestino"
```

## Resolução de problemas

### "Não consegui conectar no servidor"

- Verifique se você está na mesma rede Wi-Fi/LAN que o servidor
- Confirme o IP e o PIN com o organizador
- Teste com: `ping <IP_DO_SERVIDOR>`

### "Ollama não responde" / "Erro ao consultar Ollama"

1. Verifique se o Ollama está rodando: `ollama list`
2. Teste o modelo manualmente: `ollama run qwen3:0.6b`
3. Verifique se a porta está aberta: `curl http://localhost:11434`

### "Import langchain could not be resolved"

```bash
pip install langchain langchain-community
```

## Recursos

- [Ollama - Modelos disponíveis](https://ollama.ai/library)
- [LangChain - Documentação](https://python.langchain.com/)
- [Guia de Prompt Engineering](https://www.promptingguide.ai/)
