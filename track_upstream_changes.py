#!/usr/bin/env python3
"""
Upstream Tracking Script for COQUI_AI_TTS

This script compares our local COQUI_AI_TTS directory against the upstream
GitHub repository to identify changes, new files, and potential updates needed.

Usage:
    python track_upstream_changes.py [--baseline v0.27.1] [--target v0.27.2] [--output report.md]
"""

import argparse
import hashlib
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


class UpstreamTracker:
    """Track changes between local COQUI_AI_TTS and upstream repository."""
    
    def __init__(self, local_path: str, repo_url: str = "https://github.com/idiap/coqui-ai-TTS"):
        self.local_path = Path(local_path)
        self.repo_url = repo_url
        self.api_url = repo_url.replace("github.com", "api.github.com/repos")
        self.temp_dir = None
        
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "ERROR"
    
    def get_local_files(self) -> Dict[str, str]:
        """Get all files in local COQUI_AI_TTS with their hashes."""
        files = {}
        if not self.local_path.exists():
            return files
            
        for file_path in self.local_path.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                relative_path = file_path.relative_to(self.local_path)
                # Convert to forward slashes for consistency with GitHub
                relative_str = str(relative_path).replace(os.sep, '/')
                files[relative_str] = self.get_file_hash(file_path)
        
        return files
    
    def download_upstream_files(self, tag: str) -> Dict[str, str]:
        """Download and hash files from upstream repository at given tag."""
        print(f"📥 Downloading upstream files from {tag}...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coqui_upstream_"))
        upstream_path = self.temp_dir / "upstream"
        
        try:
            # Clone the repository at specific tag
            cmd = [
                "git", "clone", "--depth", "1", "--branch", tag,
                self.repo_url, str(upstream_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)
            
            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr}")
            
            # Get all files from TTS directory (equivalent to our COQUI_AI_TTS)
            tts_path = upstream_path / "TTS"
            files = {}
            
            if tts_path.exists():
                for file_path in tts_path.rglob("*"):
                    if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                        relative_path = file_path.relative_to(tts_path)
                        relative_str = str(relative_path).replace(os.sep, '/')
                        files[relative_str] = self.get_file_hash(file_path)
            
            return files
            
        except Exception as e:
            print(f"❌ Error downloading upstream: {e}")
            return {}
    
    def get_github_releases(self) -> List[Dict]:
        """Get recent releases from GitHub API."""
        try:
            response = requests.get(f"{self.api_url}/releases", timeout=10)
            response.raise_for_status()
            return response.json()[:10]  # Last 10 releases
        except Exception as e:
            print(f"⚠️  Could not fetch releases: {e}")
            return []
    
    def categorize_files(self, local_files: Dict[str, str], baseline_files: Dict[str, str]) -> Dict[str, List[str]]:
        """Categorize local files as unchanged, modified, or new."""
        categories = {
            "unchanged": [],
            "modified": [],
            "local_only": []
        }
        
        for file_path, local_hash in local_files.items():
            if file_path in baseline_files:
                if local_hash == baseline_files[file_path]:
                    categories["unchanged"].append(file_path)
                else:
                    categories["modified"].append(file_path)
            else:
                categories["local_only"].append(file_path)
        
        return categories
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory with Windows-compatible method."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            import stat
            
            def handle_remove_readonly(func, path, exc):
                """Handle removal of readonly files on Windows."""
                if os.path.exists(path):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
            
            try:
                shutil.rmtree(self.temp_dir, onerror=handle_remove_readonly)
            except Exception as e:
                print(f"⚠️  Warning: Could not fully clean temp directory: {e}")
    
    def compare_versions(self, baseline_tag: str, target_tag: str) -> Dict:
        """Compare two upstream versions."""
        print(f"🔍 Comparing {baseline_tag} → {target_tag}")
        
        baseline_files = self.download_upstream_files(baseline_tag)
        if not baseline_files:
            return {"error": f"Could not download {baseline_tag}"}
            
        # Clean up and re-download for target
        self.cleanup_temp_dir()
            
        target_files = self.download_upstream_files(target_tag)
        if not target_files:
            return {"error": f"Could not download {target_tag}"}
        
        # Compare files
        all_files = set(baseline_files.keys()) | set(target_files.keys())
        
        changes = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        for file_path in all_files:
            baseline_hash = baseline_files.get(file_path)
            target_hash = target_files.get(file_path)
            
            if baseline_hash is None:
                changes["added"].append(file_path)
            elif target_hash is None:
                changes["removed"].append(file_path)
            elif baseline_hash != target_hash:
                changes["modified"].append(file_path)
        
        return changes
    
    def generate_report(self, baseline_tag: str, target_tag: str, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive tracking report."""
        report_lines = [
            f"# COQUI_AI_TTS Upstream Tracking Report",
            f"",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Local Path**: `{self.local_path}`",
            f"**Baseline**: {baseline_tag}",
            f"**Target**: {target_tag}",
            f"**Repository**: {self.repo_url}",
            f"",
        ]
        
        # Get local files
        local_files = self.get_local_files()
        report_lines.extend([
            f"## 📊 Local Status",
            f"",
            f"- **Total local files**: {len(local_files)}",
            f"- **Local directory**: `{self.local_path.name}/`",
            f"",
        ])
        
        # Compare against baseline to categorize our local files
        print(f"📥 Downloading baseline {baseline_tag} for comparison...")
        baseline_files = self.download_upstream_files(baseline_tag)
        
        if baseline_files:
            categories = self.categorize_files(local_files, baseline_files)
            
            report_lines.extend([
                f"## 🔍 Local File Analysis (vs {baseline_tag})",
                f"",
                f"| Category | Count | Description |",
                f"|----------|-------|-------------|",
                f"| **Unchanged** | {len(categories['unchanged'])} | Files identical to {baseline_tag} |",
                f"| **Modified** | {len(categories['modified'])} | Files we've customized |",
                f"| **Local Only** | {len(categories['local_only'])} | Files not in upstream |",
                f"",
            ])
            
            # List modified files (our customizations)
            if categories["modified"]:
                report_lines.extend([
                    f"### 🔧 Modified Files (Our Customizations)",
                    f"",
                    f"These files differ from upstream {baseline_tag} - likely our modifications:",
                    f""
                ])
                for file_path in sorted(categories["modified"]):
                    report_lines.append(f"- `{file_path}`")
                report_lines.append("")
            
            # List local-only files  
            if categories["local_only"]:
                report_lines.extend([
                    f"### ➕ Local-Only Files",
                    f"",
                    f"These files don't exist in upstream {baseline_tag}:",
                    f""
                ])
                for file_path in sorted(categories["local_only"]):
                    report_lines.append(f"- `{file_path}`")
                report_lines.append("")
        
        # Compare upstream versions
        upstream_changes = self.compare_versions(baseline_tag, target_tag)
        
        if "error" not in upstream_changes:
            report_lines.extend([
                f"## 🔄 Upstream Changes ({baseline_tag} → {target_tag})",
                f"",
            ])
            
            # Summary table
            total_changes = sum(len(v) for v in upstream_changes.values() if isinstance(v, list))
            report_lines.extend([
                f"| Change Type | Count |",
                f"|-------------|-------|",
                f"| **Added** | {len(upstream_changes.get('added', []))} |",
                f"| **Removed** | {len(upstream_changes.get('removed', []))} |", 
                f"| **Modified** | {len(upstream_changes.get('modified', []))} |",
                f"| **Total** | {total_changes} |",
                f"",
            ])
            
            # Detailed changes
            for change_type, files in upstream_changes.items():
                if files and isinstance(files, list):
                    icon = {"added": "➕", "removed": "➖", "modified": "🔄"}[change_type]
                    report_lines.extend([
                        f"### {icon} {change_type.title()} Files",
                        f""
                    ])
                    for file_path in sorted(files):
                        # Check if this file affects us
                        affects_us = file_path in local_files
                        status = " 🚨 **AFFECTS US**" if affects_us else ""
                        report_lines.append(f"- `{file_path}`{status}")
                    report_lines.append("")
        
        # Get recent releases
        releases = self.get_github_releases()
        if releases:
            report_lines.extend([
                f"## 📋 Recent Releases",
                f"",
            ])
            for release in releases[:5]:  # Show top 5
                published = release.get('published_at', 'Unknown')[:10]  # YYYY-MM-DD
                report_lines.append(f"- **{release['tag_name']}** ({published}): {release['name']}")
            report_lines.append("")
        
        # Recommendations
        if "error" not in upstream_changes:
            affected_files = [f for f in upstream_changes.get('modified', []) + upstream_changes.get('added', []) 
                            if f in local_files]
            
            report_lines.extend([
                f"## 💡 Recommendations",
                f"",
            ])
            
            if affected_files:
                report_lines.extend([
                    f"### ⚠️ Files Needing Review",
                    f"",
                    f"These files changed upstream and exist in our local copy:",
                    f""
                ])
                for file_path in sorted(affected_files):
                    is_modified = file_path in categories.get("modified", [])
                    status = " (Our customizations may conflict)" if is_modified else " (Should be safe to update)"
                    report_lines.append(f"- `{file_path}`{status}")
                report_lines.append("")
            
            if len(upstream_changes.get('modified', [])) == 0:
                report_lines.append(f"✅ **No file conflicts** - upstream changes don't affect our local files.")
            else:
                report_lines.extend([
                    f"### 🔧 Next Steps",
                    f"",
                    f"1. Review affected files above",
                    f"2. For unmodified files: consider updating from upstream",
                    f"3. For modified files: carefully merge changes",
                    f"4. Test thoroughly after any updates",
                    f""
                ])
        
        # Clean up
        self.cleanup_temp_dir()
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Track upstream changes in COQUI_AI_TTS")
    parser.add_argument("--local-path", default="skyrimnet-xtts/COQUI_AI_TTS", 
                       help="Path to local COQUI_AI_TTS directory")
    parser.add_argument("--baseline", default="v0.27.1",
                       help="Baseline upstream tag (what we originally used)")
    parser.add_argument("--target", default="v0.27.2", 
                       help="Target upstream tag (what to compare against)")
    parser.add_argument("--output", help="Output file for report (optional)")
    parser.add_argument("--repo", default="https://github.com/idiap/coqui-ai-TTS",
                       help="Upstream repository URL")
    
    args = parser.parse_args()
    
    print(f"🔍 COQUI_AI_TTS Upstream Tracker")
    print(f"================================")
    
    tracker = UpstreamTracker(args.local_path, args.repo)
    
    try:
        report = tracker.generate_report(args.baseline, args.target, args.output)
        
        if not args.output:
            print("\n" + report)
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()