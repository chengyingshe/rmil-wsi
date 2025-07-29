# WSI Thumbnail Generator

This script generates thumbnails from Whole Slide Images (WSI) files for visualization and analysis purposes.

## Features

- Support for multiple WSI formats (SVS, TIF, NDPI, etc.)
- Automatic level selection for optimal thumbnail quality
- Batch processing with progress tracking
- CSV result logging
- Auto-skip existing thumbnails
- Customizable thumbnail size with aspect ratio preservation
- Specific WSI level selection
- Maintains original aspect ratio by scaling minimum edge to target size
- **Tumor annotation overlay** - Draw tumor regions from XML annotation files

## Requirements

```bash
pip install openslide-python pillow pandas tqdm
```

## Basic Usage

### Simple thumbnail generation:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 1024 1024
```

### Multiple WSI formats:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 1024 1024 \
    --wsi_format "svs;tif;ndpi"
```

### Custom thumbnail size:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 2048 2048
```

### Specific WSI level:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 1024 1024 \
    --level 2
```

### Process specific files from CSV:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 1024 1024 \
    --process_list "process_list.csv"
```

### With tumor annotations:
```bash
python create_thumbnails.py \
    --source /path/to/wsi/files \
    --output ./thumbnails \
    --target_size 1024 1024 \
    --annotation_dir /path/to/annotations
```

## Command Line Arguments

- `--source`: Path to folder containing WSI files (required)
- `--output`: Directory to save thumbnails (required)
- `--target_size`: Target thumbnail size (width height), default: 1024 1024
- `--level`: Specific WSI level to use for thumbnail (optional)
- `--wsi_format`: WSI file format(s), use semicolon to separate multiple formats, default: "svs"
- `--no_auto_skip`: Don't skip existing thumbnails
- `--process_list`: CSV file with specific files to process
- `--annotation_dir`: Directory containing XML annotation files

## Output

The script generates:
1. **Thumbnail images**: JPEG files named after the original WSI files
2. **Results CSV**: `thumbnail_results.csv` with processing statistics
3. **Console output**: Progress tracking and summary statistics

## Example Output Structure

```
thumbnails/
├── slide_001.jpg
├── slide_002.jpg
├── slide_003.jpg
└── thumbnail_results.csv
```

## CSV Results Format

The `thumbnail_results.csv` file contains:
- `slide_id`: Original WSI filename
- `status`: Processing status (success, failed, already_exists, failed_info)
- `time`: Processing time in seconds
- `output_path`: Path to generated thumbnail
- `has_annotations`: Whether tumor annotations were applied

## Performance Tips

1. **Use appropriate target size**: Larger thumbnails take more time to generate
2. **Specify WSI level**: Use `--level` to avoid automatic level selection overhead
3. **Batch processing**: Process multiple files together for better efficiency
4. **Auto-skip**: Use default auto-skip to avoid reprocessing existing thumbnails

## Error Handling

The script handles common errors:
- Invalid WSI files
- Missing source directories
- Permission issues
- Memory limitations

Failed files are logged in the results CSV with appropriate error status.

## Tumor Annotation Support

The script supports XML annotation files (ASAP format) for tumor region overlay:
- Automatically finds corresponding annotation files by slide ID
- Scales annotation coordinates to match thumbnail size
- Draws tumor regions with red outline and semi-transparent red fill
- Maintains aspect ratio during coordinate transformation

## Integration with RMIL Project

This thumbnail generator can be used with the RMIL project for:
- WSI visualization with tumor annotations
- Quality assessment
- Documentation
- Presentation materials

## Example Integration

```bash
# Generate thumbnails for your dataset
python create_thumbnails.py \
    --source datasets/mydatasets/TCGA-lung-uni2/ \
    --output datasets/mydatasets/TCGA-lung-uni2/thumbnails \
    --target_size 1024 1024 \
    --wsi_format "svs"
```

This will create thumbnails that can be used with the visualization tools in the RMIL project. 