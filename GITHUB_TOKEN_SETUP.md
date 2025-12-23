# GitHub Token Setup Guide

## Security Issue Resolved

GitHub automatically revoked the previous token because it was detected in the code. This is a security feature to protect your account.

## How to Set Up New Token

### 1. Create New GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Set token name: "GHCR Push Token"
4. Set expiration: Choose appropriate duration
5. Select scopes:
   - ✅ `write:packages` (to push images)
   - ✅ `read:packages` (to pull images)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again)

### 2. Set Environment Variable

```bash
# Set the token as environment variable (recommended)
export GITHUB_TOKEN=ghp_your_new_token_here

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export GITHUB_TOKEN=ghp_your_new_token_here' >> ~/.bashrc
source ~/.bashrc
```

### 3. Test the Setup

```bash
# Test Docker login
echo $GITHUB_TOKEN | docker login ghcr.io -u sergevil --password-stdin

# If successful, you can now push images
./push-to-ghcr.sh
```

## Security Best Practices

### ✅ **Do This:**
- Use environment variables for tokens
- Set appropriate token expiration
- Use minimal required scopes
- Never commit tokens to git

### ❌ **Don't Do This:**
- Hardcode tokens in scripts
- Commit tokens to repositories
- Share tokens publicly
- Use tokens with excessive permissions

## Troubleshooting

### Token Not Working
```bash
# Check if token is set
echo $GITHUB_TOKEN

# Test GitHub API
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

### Docker Login Issues
```bash
# Clear existing credentials
docker logout ghcr.io

# Try login again
echo $GITHUB_TOKEN | docker login ghcr.io -u sergevil --password-stdin
```

## Alternative: Use GitHub CLI

If you prefer, you can also use GitHub CLI:

```bash
# Install GitHub CLI
brew install gh  # macOS
# or download from: https://cli.github.com/

# Login
gh auth login

# Then use ghcr.io without explicit token
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2025i_latest
``` 