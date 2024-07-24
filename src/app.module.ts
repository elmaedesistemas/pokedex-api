import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { TensorflowModule } from './modules/tensorflow/tensorflow.module';

@Module({
  imports: [TensorflowModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
