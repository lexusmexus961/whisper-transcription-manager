# Security policy

## Supported versions

Security fixes go into the latest release and `main` only.

## Reporting a vulnerability

Please don't open a public issue. Report it privately on GitHub:
[github.com/lexusmexus961/whisper-transcription-manager/security/advisories/new](https://github.com/lexusmexus961/whisper-transcription-manager/security/advisories/new)
(the repository's **Security** tab → **Report a vulnerability**). Include the
version or commit, the relevant configuration (without passwords or tokens)
and the steps to reproduce it.

## Scope

Whisper Transcription Manager is made for a trusted network. Out of the box it listens on this
machine only and asks for no login; opened to the network without a login
(docs/GUIDE.md: Network Access), anyone who reaches it may use it — that documented
setup is not a vulnerability. In scope, for example:

- getting around the login or the API token, the origin and host check
  (`security.trusted_origins`), the folder limits (`security.allowed_roots`)
  or the MCP token;
- reading or writing files outside the configured folders;
- running commands through settings, uploads, links or the language model's
  tools;
- secrets (tokens, keys, passwords) in logs, API answers or exports;
- a web page in the user's browser making the server do something;
- text from a recording, a title, a file name or a model's answer that runs
  as script in the page or in the extension's bar on YouTube;
- transcript text that makes Ask run an action while `chat.allow_actions` is
  off, or that gets past `mcp.allow_writes` or `security.allowed_roots`.

An outside AI client that follows instructions in a transcript while work is
allowed for it is a documented limitation, not a vulnerability (docs/GUIDE.md: MCP
Server → Prompt injection). So is a link download that a public address
redirects into the local network: the private-host check reads the link's own
host only (`youtube.allow_private_hosts` in `config.default.yaml`).

Bugs in dependencies or models belong to their projects.
