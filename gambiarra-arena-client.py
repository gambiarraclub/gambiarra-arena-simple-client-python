#!/usr/bin/env python3
import asyncio
import json
import time

import websockets
import httpx

# ============================================
# CONFIGURAÇÕES - Edite aqui conforme necessário
# ============================================
HOST = "192.168.0.212"  # IP do servidor da arena
PIN = "544208"  # PIN da sessão
PARTICIPANT_ID = "meuId"  # ID do participante
NICKNAME = "Python Mock"  # Apelido do participante

# Configurações do Ollama
OLLAMA_HOST = "localhost"  # Host do servidor Ollama
OLLAMA_MODEL = "qwen3:0.6b"  # Modelo a ser usado (ex: llama3.2, mistral, etc)
# ============================================

# URLs construídas automaticamente
URL = f"ws://{HOST}:3000/ws"
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434"


def context_engineering(prompt: str) -> str:
    """
    Adiciona contexto e instruções ao prompt para melhorar a qualidade
    das respostas do modelo.
    """
    system_instructions = """You are a helpful, accurate, and concise AI assistant.
Follow these guidelines:
- Be direct and to the point
- Provide accurate information
- If you don't know something, say so
- Use clear and simple language
- Focus on answering what was asked

"""

    enhanced_prompt = system_instructions + prompt
    return enhanced_prompt


async def handle_challenge(ws, challenge: dict):
    """
    Trata mensagem de desafio vinda do servidor e envia:
      - sequência de tokens gerados pelo Ollama (streaming)
      - mensagem de conclusão com métricas
    """
    round_id = challenge.get("round")
    prompt = challenge.get("prompt", "")

    enhanced_prompt = context_engineering(prompt)

    print("\n=== Novo desafio recebido ===")
    print(f"round: {round_id}")
    print(f"prompt: {enhanced_prompt}")
    print("=============================\n")

    # Preparar requisição para Ollama
    ollama_request = {
        "model": OLLAMA_MODEL,
        "prompt": enhanced_prompt,
        "stream": True,
    }

    seq = 0
    total_tokens = 0
    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""

    print(f"Consultando Ollama (modelo: {OLLAMA_MODEL})...")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json=ollama_request,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Ollama retorna o token no campo "response"
                    token_content = chunk.get("response", "")
                    if not token_content:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    full_response += token_content
                    total_tokens += 1

                    # Enviar token para o servidor
                    token_msg = {
                        "type": "token",
                        "round": round_id,
                        "participant_id": PARTICIPANT_ID,
                        "seq": seq,
                        "content": token_content,
                    }

                    await ws.send(json.dumps(token_msg))
                    seq += 1

                    # Verificar se é o último chunk
                    if chunk.get("done", False):
                        break

    except Exception as e:
        print(f"❌ Erro ao consultar Ollama: {e}")
        return

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    latency_first_token_ms = int((first_token_time - start_time) * 1000)
    duration_ms = int((end_time - start_time) * 1000)

    print(f"\n✅ Resposta completa ({total_tokens} tokens):")
    print(full_response)
    print()

    # Enviar mensagem de conclusão
    complete_msg = {
        "type": "complete",
        "round": round_id,
        "participant_id": PARTICIPANT_ID,
        "tokens": total_tokens,
        "latency_ms_first_token": latency_first_token_ms,
        "duration_ms": duration_ms,
    }

    print("Enviando mensagem de conclusão com métricas:")
    print(json.dumps(complete_msg, indent=2, ensure_ascii=False))

    await ws.send(json.dumps(complete_msg))


async def client_loop():
    """
    Faz a conexão, registra o participante e entra no loop
    para receber mensagens do servidor.
    """

    print(f"Conectando a {URL} ...")

    async with websockets.connect(URL) as ws:
        print("✅ Conectado ao servidor WebSocket.")

        # Mensagem de registro inicial (Client → Server)
        register_msg = {
            "type": "register",
            "participant_id": PARTICIPANT_ID,
            "nickname": NICKNAME,
            "pin": PIN,
            "runner": "ollama",
            "model": OLLAMA_MODEL,
        }

        print("Enviando mensagem de registro:")
        print(json.dumps(register_msg, indent=2, ensure_ascii=False))
        await ws.send(json.dumps(register_msg))

        # Loop principal de recebimento de mensagens
        while True:
            try:
                raw_msg = await ws.recv()
            except websockets.ConnectionClosed as e:
                print(f"Conexão fechada pelo servidor: {e.code} - {e.reason}")
                break

            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print(f"Mensagem não é JSON válido: {raw_msg}")
                continue

            msg_type = data.get("type")

            if msg_type == "challenge":
                # Mensagem de desafio: gera resposta mockada
                await handle_challenge(ws, data)

            elif msg_type == "heartbeat":
                # Heartbeat do servidor — aqui só logamos
                print("💓 Heartbeat recebido do servidor.")

            else:
                print("📩 Mensagem desconhecida recebida:")
                print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    try:
        asyncio.run(client_loop())
    except KeyboardInterrupt:
        print("\nEncerrando cliente por KeyboardInterrupt.")


if __name__ == "__main__":
    main()
