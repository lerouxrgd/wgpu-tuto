use wgpu_tuto::run;

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())?;
    Ok(())
}
