from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
from mmcv.parallel import is_module_wrapper
try:
    import wandb
except Exception:
    wandb = None
import os.path as osp
import time
import os
environ = os.environ

@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)

@HOOKS.register_module()
class WandbCheckpointHook(Hook):
    """Custom hook for logging checkpoints and epochs to Weights & Biases."""

    def __init__(self, interval=1, save_artifact=True, project="VADV2X"):
        self.interval = interval
        self.save_artifact = save_artifact
        self.project = project
        self.initialized = False

    def before_run(self, runner):
        """Initialize W&B once before training starts."""
        if not self.initialized:
            os.environ["WANDB_MODE"] = "online"
            os.environ["WANDB_DISABLE_CACHE"] = "true"   # no caching
            os.environ["WANDB_DISABLE_CODE"] = "true"    # don't store code snapshot
            os.environ["WANDB_SILENT"] = "true"          # suppress local console
            os.environ["WANDB_DIR"] = "/scratch/jmeng18/tmp" # logs artifacts and losses temporarily till upload

            #wandb.init(
                #project=self.project,
                #dir = '/scratch/jmeng18/tmp',
                #name=f"{osp.basename(runner.work_dir)}_{time.strftime('%Y%m%d_%H%M%S')}",
                #config={
                    #"work_dir": runner.work_dir,
                    #"seed": getattr(runner, "seed", None),
                    #"max_epochs": getattr(runner, "max_epochs", None),
                    #"config_file": runner.meta.get("config", None) if hasattr(runner, "meta") else None,
                #},
                #dir=runner.work_dir
            #)
            #self.initialized = True
            #print(f"[W&B Hook] Initialized wandb run at {runner.work_dir}")

    #def after_train_epoch(self, runner):
        """Log metrics and checkpoint to W&B after each epoch."""
        #epoch = runner.epoch + 1
        #wandb.log({"epoch": epoch})

        # Log loss and lr info
        #if hasattr(runner, "log_buffer") and runner.log_buffer.output:
            #wandb.log(runner.log_buffer.output)

        # Log checkpoint as artifact every N epochs
        #if self.save_artifact and epoch % self.interval == 0:
            #ckpt_path = osp.join(runner.work_dir, f"epoch_{epoch}.pth")
            #if osp.exists(ckpt_path):
                #artifact = wandb.Artifact(f"vad_epoch_{epoch}", type="model")
                #artifact.add_file(ckpt_path)
                #wandb.log_artifact(artifact)
                #print(f"[W&B Hook] Logged checkpoint: {ckpt_path}")

    #def after_run(self, runner):
        """Cleanly close the wandb run."""
        #wandb.finish()
        #print("[W&B Hook] Finished wandb run.")
