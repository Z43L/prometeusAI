# prometheus_agi/main.py
import argparse
import asyncio
import multiprocessing
import os
import torch

# Importaciones de la nueva estructura
from config import SANDBOX_DIR, CUSTOM_GENES_DIR, FORGE_DIR, MIND_STATE_DIR
from core.mind import PrometheusAGI
from genes.interaction import VoiceInteractionGene
from environments.grid_world import GridWorldEnv, WorldStream
from evolution.dojo import Dojo, ConsciousnessLoop
from utils.pretraining import run_optimized_pretraining, setup_environment

def setup_directories():
    """Crea los directorios necesarios."""
    for d in [SANDBOX_DIR, CUSTOM_GENES_DIR, FORGE_DIR, MIND_STATE_DIR]:
        os.makedirs(d, exist_ok=True)

async def run_interactive_mode_async(prometheus: "PrometheusAGI"):
    """Modo interactivo asíncrono."""
    print("\n" + "="*60 + "\nINICIANDO MODO INTERACTIVO (Async). Di 'salir' para terminar.\n" + "="*60)
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nTU ORDEN > ")
            if user_input.strip().lower() in ['exit', 'quit', 'salir']: break
            ai_response = await prometheus.think(user_input)
            print("\n" + "-"*45 + f"\nPROMETEO RESPONDE: {ai_response}\n" + "-"*45)
        except KeyboardInterrupt: break

def run_voice_mode(prometheus: "PrometheusAGI"):
    """Modo interactivo por voz."""
    print("\n" + "="*60 + "\nINICIANDO MODO INTERACTIVO POR VOZ. Di 'adiós' para terminar.\n" + "="*60)
    voice_gene = VoiceInteractionGene()
    voice_gene.speak("Hola, soy Prometheus. ¿En qué puedo ayudarte?")
    while True:
        try:
            user_input = voice_gene.listen()
            if not user_input: continue
            if user_input.lower() in ['adiós', 'salir']:
                voice_gene.speak("Hasta luego.")
                break
            ai_response = asyncio.run(prometheus.think(user_input))
            voice_gene.speak(ai_response or "No he generado una respuesta textual.")
        except KeyboardInterrupt: break

def main(args):
    """Función principal que orquesta la ejecución de Prometheus."""
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Advertencia: No se detectó GPU. Se usará CPU.")

    setup_directories()
    setup_environment() # Crea el dataset de ejemplo si no existe

    prometheus_mind = None
    try:
        # Los modos de pre-entrenamiento no necesitan una instancia completa a largo plazo
        if args.mode == 'pretrain':
            # Creamos una instancia temporal solo para el pre-entrenamiento
            temp_mind = PrometheusAGI(force_genesis=True)
            run_optimized_pretraining(temp_mind, batch_size=100)
            temp_mind.shutdown()
            return
        
        # Para los otros modos, creamos la instancia principal
        prometheus_mind = PrometheusAGI(force_genesis=args.force_genesis)

        if args.mode == 'interactive':
            asyncio.run(run_interactive_mode_async(prometheus_mind))
        elif args.mode == 'voice':
            run_voice_mode(prometheus_mind)
        elif args.mode == 'marathon':
            prometheus_mind.dojo.run_marathon(generations=args.generations, population_size=args.population)
        elif args.mode == 'simulation':
            env = GridWorldEnv()
            world_stream = WorldStream(environment=env)
            consciousness_loop = ConsciousnessLoop(prometheus_mind, world_stream)
            prometheus_mind.env_size = env.size
            try:
                world_stream.start()
                consciousness_loop.run()
            finally:
                world_stream.stop()
                world_stream.join(timeout=2)

    except Exception as e:
        print(f"\n[ERROR FATAL] Ocurrió un error en la ejecución: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if prometheus_mind:
            prometheus_mind.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PrometheusAGI - Una IA Evolutiva y Explicable")
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'voice', 'marathon', 'simulation', 'pretrain'],
                        help='Modo de ejecución.')
    parser.add_argument('--population', type=int, default=15, help='Tamaño de la población para la evolución.')
    parser.add_argument('--generations', type=int, default=8, help='Número de generaciones por evolución.')
    parser.add_argument('--force-genesis', action='store_true', help='Ignora el estado guardado y empieza de cero.')
    
    # Es crucial para que multiprocessing funcione correctamente en todos los sistemas operativos
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parser.parse_args()
    main(args)