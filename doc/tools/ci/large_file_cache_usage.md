# Large File Cache (LFC) Usage Guide

## Overview

Large File Cache (LFC) is a service provided by GitHub CI infrastructure team that serves as an HTTP file server populated by the contents of a common S3 bucket. It provides a convenient way to store and access large files that are needed for testing and development but should not be stored directly in Git repositories.

## What is Large File Cache?

LFC is designed to solve the problem of managing large files in software development workflows:

- **Purpose**: Store large test files, models, and datasets outside of Git repositories
- **Implementation**: HTTP file server backed by S3 storage
- **Access**: Available through CI actions and TT systems behind Tailscale
- **Permissions**: Files are readable by all within TT organization

> **Security Note**: LFC should NOT be used for files with restricted access requirements, as all files are readable to everyone within the organization.

## Access Methods

### CI Actions
LFC is directly accessible from GitHub Actions workflows and CI pipelines through internal network paths.

### TT Systems (Tailscale)
LFC can be accessed from TT development systems that are connected through Tailscale VPN, though through different network paths than CI.

### Command Line Access
The recommended way to access LFC files is through the `lfc_downloader.sh` utility script located in `tools/ci/lfc_downloader.sh`.

## File Organization Strategy

### Gzipped Tar Archives
While LFC can host arbitrary directory structures, the current convention *we use* is to store related files as gzipped tar archives (`.tar.gz`). This approach offers several benefits:

- **Fewer Objects**: Reduces the number of individual objects stored in LFC
- **Atomic Downloads**: Ensures all related files are downloaded together
- **Compression**: Reduces storage space and download time
- **Automatic Extraction**: The `lfc_downloader.sh` script handles extraction automatically

### Directory Naming Convention

Files extracted from LFC archives follow a specific naming convention:

- **Pattern**: Directories start with double underscores (`__`)
- **Purpose**: Ensures Git ignores these directories automatically
- **Examples**: `tests/__models`, `tests/__data_files`

This convention provides several advantages:
- Git automatically ignores these directories (via `.gitignore`)
- Prevents accidental check-in of large files to repositories
- Clearly identifies LFC-sourced content
- Maintains clean repository structure

> **Note**: The double underscore convention is a project-specific best practice, not a requirement of LFC or the downloader script.

## Current LFC Contents

### Available Archives

| Archive Name | Description | Extraction Path | Use Case |
|--------------|-------------|-----------------|----------|
| `test_onnx_models.tar.gz` | ONNX model files for testing | `tests/__models/` | Unit tests for ONNX path |
| `llk_elf_files.tar.gz` | LLK front-end test files | `tests/__data_files/` | LLK front-end testing |

### Extraction Behavior

All archives are designed to be extracted in the repository root directory:

```bash
# When extracted from repo root:
test_onnx_models.tar.gz → tests/__models/
llk_elf_files.tar.gz   → tests/__data_files/
```

## Using the LFC Downloader

### Basic Usage

```bash
# Download and extract ONNX models
./tools/ci/lfc_downloader.sh --extract test_onnx_models.tar.gz

# Download and extract LLK ELF files
./tools/ci/lfc_downloader.sh --extract llk_elf_files.tar.gz
```

### Advanced Options

```bash
# Dry run to see what would be downloaded
./tools/ci/lfc_downloader.sh --dryrun --extract test_onnx_models.tar.gz

# Verbose output for debugging
./tools/ci/lfc_downloader.sh --verbose --extract llk_elf_files.tar.gz

# CI mode (automatically detected in GitHub Actions)
./tools/ci/lfc_downloader.sh --extract test_onnx_models.tar.gz
```

For complete documentation of the downloader script, see [lfc_downloader_user_guide.md](lfc_downloader_user_guide.md).

## Uploading Files to LFC

### Adding New Files to LFC

To upload new files or archives to the Large File Cache, you need to contact the project maintainers. The LFC service is managed centrally to ensure proper organization and prevent conflicts.

#### Process for Uploading

1. **Prepare Your Files**
   - Create a gzipped tar archive (`.tar.gz`) of related files
   - Choose a descriptive name for your archive
   - Ensure the archive extracts to a directory starting with `__` (following project convention)
   - Test the archive locally to verify correct extraction

2. **Contact Maintainers**
   - **Primary Contact**: Project maintainers
   - **Information to Provide**:
     - Archive file name and purpose
     - Intended extraction path
     - Description of contents
     - Use case and justification for LFC storage
     - Expected file size

3. **Upload Coordination**
   - Maintainers will coordinate with the GitHub CI infrastructure team
   - Upload will be performed through proper channels to the S3 backend
   - Archive will be tested for accessibility through both CI and Tailscale paths

#### Requirements for New Archives

- **File Size**: Should be large enough to justify LFC storage (typically > 10MB)
- **Access Pattern**: Files should be accessed by multiple team members or CI processes
- **Stability**: Files should not change frequently
- **Security**: Files must not contain sensitive or restricted content
- **Organization**: Related files should be grouped into logical archives

#### Example Request

```
Subject: LFC Upload Request - Deep Learning Model Files

Hi [Maintainer],

I would like to upload a new archive to LFC:

Archive Name: dl_pretrained_models.tar.gz
Size: ~150MB
Purpose: Pre-trained deep learning models for integration tests
Extraction Path: tests/__dl_models/
Contents: 3 PyTorch model files (.pth) and corresponding config files
Justification: Required for ML pipeline tests, too large for Git repository

Please let me know if you need any additional information.

Thanks,
[Your Name]
```

### Updating Existing Archives

For updates to existing LFC archives:

1. Contact maintainers with details of changes needed
2. Provide the updated archive file
3. Specify if this is a replacement or addition
4. Include migration notes if the update affects existing workflows

> **Important**: Always coordinate with maintainers before making changes to ensure proper version management and team communication.

## Best Practices

### When to Use LFC

✅ **Good Use Cases:**
- Large test datasets
- Pre-trained model files
- Binary test fixtures
- Large reference files for validation
- Files that change infrequently
- Files that would significantly increase repository size

❌ **Avoid LFC for:**
- Sensitive or restricted access files
- Files that change frequently with code
- Small configuration files
- Source code or documentation
- Files requiring version control history

### File Management

1. **Archive Related Files**: Group logically related files into single tar.gz archives
2. **Use Descriptive Names**: Choose clear, descriptive names for archives
3. **Follow Naming Convention**: Extract to directories starting with `__`
4. **Document Contents**: Maintain documentation of what each archive contains
5. **Regular Cleanup**: Periodically review and remove obsolete files

### Git Integration

1. **Update .gitignore**: Ensure extraction paths are in `.gitignore`
2. **Document Dependencies**: Note LFC dependencies in project documentation
3. **CI Integration**: Include LFC downloads in CI/CD pipelines as needed

## Workflow Examples

### Development Workflow

```bash
# Clone repository
git clone <repository-url>
cd <repository>

# Download required test files
./tools/ci/lfc_downloader.sh --extract test_onnx_models.tar.gz
./tools/ci/lfc_downloader.sh --extract llk_elf_files.tar.gz

# Run tests (files are now available)
pytest tests/
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Download LFC test files
  run: |
    ./tools/ci/lfc_downloader.sh --extract test_onnx_models.tar.gz
    ./tools/ci/lfc_downloader.sh --extract llk_elf_files.tar.gz

- name: Run tests
  run: pytest tests/
```

## Troubleshooting

### Common Issues

1. **Files Not Found After Extraction**
   - Verify you're running from repository root
   - Check that `.gitignore` isn't hiding the files from your editor
   - Use `ls -la tests/` to see hidden directories

2. **Download Failures**
   - Check network connectivity
   - Verify CI mode is automatically detected (GITHUB_ACTIONS=true in CI environments)
   - Try verbose mode (`--verbose`) for detailed error information

3. **Permission Issues**
   - Ensure you have read access to the repository
   - Check that you're connected through appropriate network (Tailscale for local)

### Getting Help

- **LFC Service Issues**: Contact GitHub CI infrastructure team
- **Script Issues**: See [lfc_downloader_user_guide.md](lfc_downloader_user_guide.md)
- **Project-specific Issues**: Check project documentation or contact maintainers

## Reference Links

- [Large File Cache Service Documentation](https://tenstorrent.atlassian.net/wiki/spaces/CI/pages/1089307133/Large+File+Cache)
- [LFC Downloader Script Documentation](lfc_downloader_user_guide.md)

## Migration from Git LFS

If you're migrating from Git LFS or considering alternatives:

1. **Identify Large Files**: Use `git lfs track` output or repository analysis tools
2. **Group Related Files**: Create logical archives for related files
3. **Upload to LFC**: Coordinate with CI team to upload archives
4. **Update Workflows**: Replace Git LFS commands with LFC downloader calls
5. **Update Documentation**: Document the new download requirements

This approach can significantly reduce repository clone times and storage requirements while maintaining easy access to necessary large files.
