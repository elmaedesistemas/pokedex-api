import { Module } from '@nestjs/common';
import { TensorflowController } from './tensorflow.controller';
import { TensorflowService } from './tensorflow.service';

@Module({
  controllers: [TensorflowController],
  providers: [TensorflowService],
  exports: [TensorflowService],
})
export class TensorflowModule {}
