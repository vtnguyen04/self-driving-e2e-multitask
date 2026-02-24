import argparse
import sys
from pathlib import Path
import logging
from neuro_pilot.utils.logger import logger

# Ensure root is in path if running from here (packaging handles this usually, but for dev...)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


logger = logging.getLogger(__name__)

from neuro_pilot.engine.model import NeuroPilot

def main():
    parser = argparse.ArgumentParser(description="NeuroPilot MultiTask Training")
    parser.add_argument('model', type=str, nargs='?', help='Model config file (yaml) or weights (pt)')
    parser.add_argument('--task', type=str, default=None, help='Task name (e.g. multitask, detect)')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--list-tasks', action='store_true', help='List available registered tasks')
    args = parser.parse_args()

    # Dynamic Task Discovery
    from neuro_pilot.engine.task import TaskRegistry

    if args.list_tasks:
        tasks = TaskRegistry.list_tasks()
        print(f"Available Tasks: {tasks}")
        return

    if not args.model:
        parser.print_help()
        return

    # Check for composite task
    task_arg = args.task
    if task_arg and ',' in task_arg:
        # Composite task!
        # We need to register it on the fly or handle it in NeuroPilot
        # NeuroPilot init calls _init_task.
        # Let's handle it by defining a temporary name or passing the list
        sub_tasks = task_arg.split(',')
        # Register a new CompositeTask under this name
        composite_name = f"composite_{'_'.join(sub_tasks)}"

        from neuro_pilot.engine.composite import CompositeTask

        @TaskRegistry.register(composite_name)
        class DynamicComposite(CompositeTask):
            def __init__(self, cfg, overrides=None, backbone=None):
                super().__init__(cfg, overrides, sub_tasks=sub_tasks)

        task_arg = composite_name
        print(f"Registered Dynamic Composite Task: {task_arg}")

    model = NeuroPilot(args.model, task=task_arg)
    model.train(max_epochs=args.epochs, batch_size=args.batch)

if __name__ == "__main__":
    main()
