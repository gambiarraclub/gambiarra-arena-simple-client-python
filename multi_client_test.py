#!/usr/bin/env python3
"""
Teste de servidor com múltiplos clientes simultâneos.
Cada cliente se conecta independentemente e responde aos desafios.
"""
import asyncio
import json
import time
import argparse

import websockets
import httpx

# ============================================
# CONFIGURAÇÕES PADRÃO
# ============================================
DEFAULT_HOST = "192.168.1.99"
DEFAULT_PIN = "715661"
DEFAULT_NUM_CLIENTS = 3
DEFAULT_OLLAMA_HOST = "localhost"
DEFAULT_OLLAMA_MODEL = "qwen3:0.6b"
# ============================================


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
    return system_instructions + prompt


class ArenaClient:
    """Cliente individual para a arena."""

    def __init__(
        self,
        client_id: int,
        host: str,
        pin: str,
        ollama_host: str,
        ollama_model: str,
    ):
        self.client_id = client_id
        self.participant_id = f"client_{client_id}"
        self.nickname = f"Python Client {client_id}"
        self.url = f"ws://{host}:3000/ws"
        self.pin = pin
        self.ollama_url = f"http://{ollama_host}:11434"
        self.ollama_model = ollama_model

    def log(self, message: str):
        """Log com prefixo do cliente."""
        print(f"[Client {self.client_id}] {message}")

    async def handle_challenge(self, ws, challenge: dict):
        """
        Trata mensagem de desafio vinda do servidor e envia:
          - sequência de tokens gerados pelo Ollama (streaming)
          - mensagem de conclusão com métricas
        """
        round_id = challenge.get("round")
        prompt = challenge.get("prompt", "")

        enhanced_prompt = context_engineering(prompt)

        self.log(f"Desafio recebido - round: {round_id}")
        self.log(f"Prompt: {prompt[:50]}...")

        ollama_request = {
            "model": self.ollama_model,
            "prompt": enhanced_prompt,
            "stream": True,
        }

        seq = 0
        total_tokens = 0
        start_time = time.perf_counter()
        first_token_time = None
        full_response = ""

        self.log(f"Consultando Ollama (modelo: {self.ollama_model})...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
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

                        token_content = chunk.get("response", "")
                        if not token_content:
                            continue

                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        full_response += token_content
                        total_tokens += 1

                        token_msg = {
                            "type": "token",
                            "round": round_id,
                            "participant_id": self.participant_id,
                            "seq": seq,
                            "content": token_content,
                        }

                        await ws.send(json.dumps(token_msg))
                        seq += 1

                        if chunk.get("done", False):
                            break

        except Exception as e:
            self.log(f"Erro ao consultar Ollama: {e}")
            return

        end_time = time.perf_counter()

        if first_token_time is None:
            first_token_time = end_time

        latency_first_token_ms = int((first_token_time - start_time) * 1000)
        duration_ms = int((end_time - start_time) * 1000)

        self.log(f"Resposta completa ({total_tokens} tokens, {duration_ms}ms)")

        complete_msg = {
            "type": "complete",
            "round": round_id,
            "participant_id": self.participant_id,
            "tokens": total_tokens,
            "latency_ms_first_token": latency_first_token_ms,
            "duration_ms": duration_ms,
        }

        await ws.send(json.dumps(complete_msg))

    async def run(self):
        """
        Faz a conexão, registra o participante e entra no loop
        para receber mensagens do servidor.
        """
        self.log(f"Conectando a {self.url}...")

        try:
            async with websockets.connect(self.url) as ws:
                self.log("Conectado ao servidor WebSocket.")

                register_msg = {
                    "type": "register",
                    "participant_id": self.participant_id,
                    "nickname": self.nickname,
                    "pin": self.pin,
                    "runner": "ollama",
                    "model": self.ollama_model,
                }

                self.log(f"Registrando como '{self.nickname}'...")
                await ws.send(json.dumps(register_msg))

                while True:
                    try:
                        raw_msg = await ws.recv()
                    except websockets.ConnectionClosed as e:
                        self.log(f"Conexão fechada: {e.code} - {e.reason}")
                        break

                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        self.log(f"Mensagem não é JSON válido: {raw_msg}")
                        continue

                    msg_type = data.get("type")

                    if msg_type == "challenge":
                        await self.handle_challenge(ws, data)
                    elif msg_type == "heartbeat":
                        self.log("Heartbeat recebido")
                    else:
                        self.log(f"Mensagem recebida: {msg_type}")

        except Exception as e:
            self.log(f"Erro na conexão: {e}")


async def run_multiple_clients(
    num_clients: int,
    host: str,
    pin: str,
    ollama_host: str,
    ollama_model: str,
):
    """Executa múltiplos clientes simultaneamente."""
    print(f"\n{'='*50}")
    print(f"Iniciando {num_clients} clientes simultâneos")
    print(f"Servidor: {host}")
    print(f"PIN: {pin}")
    print(f"Ollama: {ollama_host} (modelo: {ollama_model})")
    print(f"{'='*50}\n")

    clients = [
        ArenaClient(
            client_id=i + 1,
            host=host,
            pin=pin,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
        )
        for i in range(num_clients)
    ]

    # Executa todos os clientes simultaneamente
    tasks = [asyncio.create_task(client.run()) for client in clients]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("\nCancelando clientes...")
        for task in tasks:
            task.cancel()


def main():
    parser = argparse.ArgumentParser(
        description="Teste de servidor com múltiplos clientes simultâneos"
    )
    parser.add_argument(
        "-n", "--num-clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help=f"Número de clientes simultâneos (padrão: {DEFAULT_NUM_CLIENTS})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"IP do servidor da arena (padrão: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--pin",
        type=str,
        default=DEFAULT_PIN,
        help=f"PIN da sessão (padrão: {DEFAULT_PIN})"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help=f"Host do servidor Ollama (padrão: {DEFAULT_OLLAMA_HOST})"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Modelo Ollama (padrão: {DEFAULT_OLLAMA_MODEL})"
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_multiple_clients(
                num_clients=args.num_clients,
                host=args.host,
                pin=args.pin,
                ollama_host=args.ollama_host,
                ollama_model=args.ollama_model,
            )
        )
    except KeyboardInterrupt:
        print("\nEncerrando por KeyboardInterrupt.")


if __name__ == "__main__":
    main()
