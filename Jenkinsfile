// Jenkins pipeline for spam-detector
// - Build Docker image (model is baked in)
// - Push to registry.manty.co.kr
// - Deploy to k8s via kubectl + kustomize

pipeline {
  agent any

  options {
    disableConcurrentBuilds()
    timeout(time: 30, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '5'))
    timestamps()
  }

  environment {
    REGISTRY       = 'registry.manty.co.kr'
    IMAGE_NAME     = 'spam-detector'
    IMAGE          = "${REGISTRY}/${IMAGE_NAME}"
    K8S_NAMESPACE  = 'spam-detector'
  }

  stages {

    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.GIT_SHORT = sh(
            script: "git rev-parse --short HEAD",
            returnStdout: true
          ).trim()
          env.IMAGE_TAG = env.GIT_SHORT
          echo "Image tag: ${env.IMAGE}:${env.IMAGE_TAG}"
        }
      }
    }

    stage('Verify model artifacts') {
      steps {
        sh '''
          set -e
          test -f training/artifacts/spam_detector.onnx \
            || { echo "ERROR: training/artifacts/spam_detector.onnx 가 레포에 없습니다."; exit 1; }
          test -f training/artifacts/vocab.json \
            || { echo "ERROR: training/artifacts/vocab.json 가 레포에 없습니다."; exit 1; }
          ls -l training/artifacts/
        '''
      }
    }

    stage('Docker build') {
      steps {
        sh '''
          set -e
          docker build \
            -t ${IMAGE}:${IMAGE_TAG} \
            -t ${IMAGE}:latest \
            .
        '''
      }
    }

    stage('Docker push') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'docker-registry-credentials',
          usernameVariable: 'REG_USER',
          passwordVariable: 'REG_PASS'
        )]) {
          sh '''
            set -e
            echo "$REG_PASS" | docker login ${REGISTRY} -u "$REG_USER" --password-stdin
            docker push ${IMAGE}:${IMAGE_TAG}
            docker push ${IMAGE}:latest
            docker logout ${REGISTRY}
          '''
        }
      }
    }

    stage('Ensure registry pull secret') {
      steps {
        withCredentials([
          usernamePassword(credentialsId: 'docker-registry-credentials',
                           usernameVariable: 'REG_USER',
                           passwordVariable: 'REG_PASS'),
          file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')
        ]) {
          sh '''
            set -e
            # Namespace 가 없으면 먼저 만든다 (이후 secret 을 그 안에 넣음)
            kubectl apply -f k8s/namespace.yaml

            # 없으면 생성, 있으면 갱신 — 멱등
            kubectl -n ${K8S_NAMESPACE} create secret docker-registry registry-manty \
              --docker-server=${REGISTRY} \
              --docker-username="$REG_USER" \
              --docker-password="$REG_PASS" \
              --docker-email=manty@manty.co.kr \
              --dry-run=client -o yaml | kubectl apply -f -
          '''
        }
      }
    }

    stage('Deploy to k8s') {
      steps {
        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
          sh '''
            set -e
            cd k8s

            # kustomize 로 이미지 태그 주입
            if command -v kustomize >/dev/null 2>&1; then
              kustomize edit set image ${IMAGE}=${IMAGE}:${IMAGE_TAG}
              kustomize build . | kubectl apply -f -
            else
              # kustomize 바이너리가 없으면 kubectl 내장 사용
              kubectl kustomize . \
                | sed "s|${IMAGE}:latest|${IMAGE}:${IMAGE_TAG}|g" \
                | kubectl apply -f -
            fi

            kubectl -n ${K8S_NAMESPACE} rollout status deployment/spam-detector --timeout=180s
            kubectl -n ${K8S_NAMESPACE} get pods -l app.kubernetes.io/name=spam-detector -o wide
          '''
        }
      }
    }
  }

  post {
    success {
      echo "Deployed ${IMAGE}:${IMAGE_TAG} to namespace ${K8S_NAMESPACE}"
    }
    failure {
      echo "Pipeline failed at stage: ${env.STAGE_NAME}"
    }
    always {
      sh 'docker image prune -f || true'
    }
  }
}
