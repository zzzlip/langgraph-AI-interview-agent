FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY java-platform/pom.xml .
COPY java-platform/platform-domain/pom.xml platform-domain/
COPY java-platform/platform-application/pom.xml platform-application/
COPY java-platform/platform-interface/pom.xml platform-interface/
COPY java-platform/platform-infrastructure/pom.xml platform-infrastructure/
COPY java-platform/platform-bootstrap/pom.xml platform-bootstrap/
COPY java-platform/platform-architecture-test/pom.xml platform-architecture-test/
RUN mvn dependency:go-offline -B

COPY java-platform/ .
COPY frontend/static/ platform-bootstrap/src/main/resources/static/
RUN mvn clean package -B -DskipTests -pl platform-bootstrap -am

FROM eclipse-temurin:21-jre
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/platform-bootstrap/target/*.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]
