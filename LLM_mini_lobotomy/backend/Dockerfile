FROM rust:1-bookworm AS build
WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=build /app/target/release/lobotomy-backend /usr/local/bin/lobotomy-backend
EXPOSE 8080
CMD ["lobotomy-backend"]
