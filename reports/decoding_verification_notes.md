# Decoding strategy verification

Run the four configured decoding strategies from the repository root:

```powershell
python -m scripts.verify_decoding
```

The script prints the effective generation configuration and generated response for each strategy. Its logs record strategy start and completion without duplicating response text. Confirm that all four strategies complete and that sampling parameters are present only for sampling strategies.