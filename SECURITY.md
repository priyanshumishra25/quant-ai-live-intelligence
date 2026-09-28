# Security

## Secrets

Never commit:

- `.env`;
- Alpha Vantage keys;
- Reddit client secrets/tokens;
- JWT secrets;
- private market-data credentials;
- private datasets.

The repository ships only `.env.example` placeholders.

## Deployment credentials

The local `demo / quantai-demo` account exists only for local demonstration. Before any public deployment:

- change `AUTH_USERNAME` and `AUTH_PASSWORD`;
- replace `JWT_SECRET` with a long random secret;
- terminate TLS at a trusted reverse proxy/load balancer;
- place provider credentials in a proper secrets manager rather than Compose environment files;
- use gateway/distributed rate limiting for multi-replica deployments.

## Provider-key containment

Alpha Vantage and Reddit credentials are server-side only. The TypeScript client calls the FastAPI backend and never receives provider access credentials.

## Third-party content

Live news/Reddit text is untrusted input. The dashboard escapes third-party strings before HTML insertion and only allows `http`/`https` source links.

The application returns short snippets rather than mirroring full third-party bodies.

## Reddit data

The reference implementation processes Reddit content ephemerally for the requested analysis. Reddit content is never inserted into the historical fitting matrix and no model coefficient is learned from it; current Reddit sentiment can affect a forecast only through the documented fixed, bounded inference-time overlay. The service does not build a persistent Reddit corpus or Reddit-trained model artefact. Review current Reddit developer/data terms and obtain any required permissions before changing this boundary or deploying commercially.

## Reporting

If publishing the project, add a maintainer security contact or repository private-vulnerability-reporting mechanism before inviting external reports.
